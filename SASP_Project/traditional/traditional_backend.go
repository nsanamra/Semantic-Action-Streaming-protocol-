package main

// traditional_backend.go — Simple full-frame streaming backend
//
// Receives UDP chunks from traditional_streamer.py
// Reassembles frames, broadcasts over WebSocket
// Exposes /stats with REAL per-second bandwidth numbers
//
// Run:   go run traditional_backend.go
// View:  http://localhost:8081

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"log"
	"math"
	"net"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// ─────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────

const (
	UDPListenAddr  = "127.0.0.1:5001"
	HTTPListenAddr = ":8081"

	// Header layout (16 bytes):
	// FrameID(4) + SeqNum(2) + TotalParts(2) + Timestamp_ns(8)
	HeaderSize = 16

	BroadcastBuf = 32
)

// ─────────────────────────────────────────────
//  Metrics — all rates are per-second, reset
//  every second so numbers are always fresh.
// ─────────────────────────────────────────────

type metrics struct {
	// accumulators — written atomically from UDP goroutine
	udpFramesIn  uint64 // reassembled frames received this window
	udpBytesIn   uint64 // raw UDP bytes received this window
	wsFramesOut  uint64 // frames pushed to WebSocket this window
	wsBytesOut   uint64 // bytes pushed to WebSocket this window
	droppedTotal uint64 // cumulative dropped frames (slow clients)

	// latency samples collected this window
	latMu   sync.Mutex
	latSamp []float64

	// snapshot — written by ticker, read by /stats
	snapMu sync.RWMutex
	snap   Snapshot

	activeClients int32
	startTime     time.Time
}

// Snapshot holds one second of computed rates — safe to JSON-encode.
type Snapshot struct {
	// rates (per second)
	FPSIn  float64 `json:"fps_in"`
	FPSOut float64 `json:"fps_out"`

	// bandwidth in KB/s — computed from actual bytes transferred
	BWInKBs  float64 `json:"bandwidth_in_kbs"`
	BWOutKBs float64 `json:"bandwidth_out_kbs"`

	// latency percentiles in ms (edge timestamp → Go receive)
	LatP50 float64 `json:"latency_p50_ms"`
	LatP95 float64 `json:"latency_p95_ms"`
	LatP99 float64 `json:"latency_p99_ms"`

	// misc
	DroppedFrames uint64  `json:"dropped_frames"`
	ActiveClients int32   `json:"active_clients"`
	UptimeSec     float64 `json:"uptime_seconds"`

	// for comparison page
	Mode        string `json:"mode"`
	JpegQuality int    `json:"jpeg_quality"`
}

var m = &metrics{startTime: time.Now()}

func (m *metrics) recordUDPIn(n int) {
	atomic.AddUint64(&m.udpFramesIn, 1)
	atomic.AddUint64(&m.udpBytesIn, uint64(n))
}

func (m *metrics) recordWSOut(n int) {
	atomic.AddUint64(&m.wsFramesOut, 1)
	atomic.AddUint64(&m.wsBytesOut, uint64(n))
}

func (m *metrics) recordDrop() {
	atomic.AddUint64(&m.droppedTotal, 1)
}

func (m *metrics) recordLatency(edgeNs uint64) {
	if edgeNs == 0 {
		return
	}
	nowNs := uint64(time.Now().UnixNano())
	if edgeNs > nowNs {
		return // clock skew
	}
	ms := float64(nowNs-edgeNs) / 1e6
	if ms > 5000 {
		return // sanity clamp
	}
	m.latMu.Lock()
	m.latSamp = append(m.latSamp, ms)
	m.latMu.Unlock()
}

// tickerLoop runs every second, drains accumulators into a fresh Snapshot.
// This guarantees bandwidth numbers are always KB/s (not cumulative totals).
func (m *metrics) tickerLoop() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for range ticker.C {
		// Swap accumulators atomically
		framesIn := atomic.SwapUint64(&m.udpFramesIn, 0)
		bytesIn := atomic.SwapUint64(&m.udpBytesIn, 0)
		framesOut := atomic.SwapUint64(&m.wsFramesOut, 0)
		bytesOut := atomic.SwapUint64(&m.wsBytesOut, 0)

		// Grab and reset latency samples
		m.latMu.Lock()
		samp := make([]float64, len(m.latSamp))
		copy(samp, m.latSamp)
		m.latSamp = m.latSamp[:0]
		m.latMu.Unlock()

		p50, p95, p99 := percentiles(samp)

		snap := Snapshot{
			FPSIn:         float64(framesIn),
			FPSOut:        float64(framesOut),
			BWInKBs:       float64(bytesIn) / 1024.0, // bytes this second → KB/s
			BWOutKBs:      float64(bytesOut) / 1024.0,
			LatP50:        p50,
			LatP95:        p95,
			LatP99:        p99,
			DroppedFrames: atomic.LoadUint64(&m.droppedTotal),
			ActiveClients: atomic.LoadInt32(&m.activeClients),
			UptimeSec:     time.Since(m.startTime).Seconds(),
			Mode:          "traditional",
			JpegQuality:   80,
		}

		m.snapMu.Lock()
		m.snap = snap
		m.snapMu.Unlock()
	}
}

func (m *metrics) getSnap() Snapshot {
	m.snapMu.RLock()
	defer m.snapMu.RUnlock()
	return m.snap
}

func percentiles(s []float64) (p50, p95, p99 float64) {
	n := len(s)
	if n == 0 {
		return 0, 0, 0
	}
	sort.Float64s(s)
	pct := func(p float64) float64 {
		idx := int(math.Ceil(p/100.0*float64(n))) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= n {
			idx = n - 1
		}
		return math.Round(s[idx]*100) / 100
	}
	return pct(50), pct(95), pct(99)
}

// ─────────────────────────────────────────────
//  Reassembler
// ─────────────────────────────────────────────

type frameBuf struct {
	parts     [][]byte
	received  int
	total     int
	timestamp uint64
	createdAt time.Time
}

type Reassembler struct {
	mu     sync.Mutex
	frames map[uint32]*frameBuf
}

func newReassembler() *Reassembler {
	r := &Reassembler{frames: make(map[uint32]*frameBuf)}
	go r.gc()
	return r
}

func (r *Reassembler) add(id uint32, seq, total int, data []byte, ts uint64) ([]byte, uint64, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	fb, ok := r.frames[id]
	if !ok {
		fb = &frameBuf{
			parts:     make([][]byte, total),
			total:     total,
			timestamp: ts,
			createdAt: time.Now(),
		}
		r.frames[id] = fb
	}
	if seq >= 0 && seq < len(fb.parts) && fb.parts[seq] == nil {
		fb.parts[seq] = data
		fb.received++
	}
	if fb.received == fb.total {
		full := bytes.Join(fb.parts, nil)
		ts := fb.timestamp
		delete(r.frames, id)
		return full, ts, true
	}
	return nil, 0, false
}

func (r *Reassembler) gc() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		r.mu.Lock()
		for id, fb := range r.frames {
			if time.Since(fb.createdAt) > 2*time.Second {
				delete(r.frames, id)
				log.Printf("[reassembler] evicted stale frame %d", id)
			}
		}
		r.mu.Unlock()
	}
}

// ─────────────────────────────────────────────
//  WebSocket Hub
// ─────────────────────────────────────────────

type client struct {
	conn      *websocket.Conn
	send      chan []byte
	closeOnce sync.Once
}

func (c *client) writePump() {
	defer c.conn.Close()
	for frame := range c.send {
		if err := c.conn.WriteMessage(websocket.BinaryMessage, frame); err != nil {
			return
		}
	}
}

type Hub struct {
	mu         sync.RWMutex
	clients    map[*client]struct{}
	register   chan *client
	unregister chan *client
	broadcast  chan []byte
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*client]struct{}),
		register:   make(chan *client, 8),
		unregister: make(chan *client, 8),
		broadcast:  make(chan []byte, BroadcastBuf),
	}
}

func (h *Hub) run() {
	for {
		select {
		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = struct{}{}
			h.mu.Unlock()
			atomic.AddInt32(&m.activeClients, 1)
			log.Printf("[hub] client connected (total=%d)", atomic.LoadInt32(&m.activeClients))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				c.closeOnce.Do(func() { close(c.send) })
			}
			h.mu.Unlock()
			atomic.AddInt32(&m.activeClients, -1)
			log.Printf("[hub] client disconnected (total=%d)", atomic.LoadInt32(&m.activeClients))

		case frame := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- frame:
				default:
					m.recordDrop()
				}
			}
			h.mu.RUnlock()
		}
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin:     func(r *http.Request) bool { return true },
	ReadBufferSize:  1024,
	WriteBufferSize: 64 * 1024,
}

func serveWS(hub *Hub, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[ws] upgrade: %v", err)
		return
	}
	c := &client{conn: conn, send: make(chan []byte, 8)}
	hub.register <- c
	go c.writePump()
	go func() {
		defer func() { hub.unregister <- c }()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}()
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

func main() {
	hub := newHub()
	reassembler := newReassembler()

	go hub.run()
	go m.tickerLoop()

	// ── HTTP ─────────────────────────────────
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(indexHTML))
	})

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		serveWS(hub, w, r)
	})

	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		snap := m.getSnap()
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
			"data":   snap,
		})
	})

	http.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"healthy": true,
			"mode":    "traditional",
			"uptime":  time.Since(m.startTime).Seconds(),
		})
	})

	go func() {
		log.Printf("[http] Traditional backend → http://localhost%s", HTTPListenAddr)
		if err := http.ListenAndServe(HTTPListenAddr, nil); err != nil {
			log.Fatalf("[http] %v", err)
		}
	}()

	// ── UDP ──────────────────────────────────
	addr, err := net.ResolveUDPAddr("udp", UDPListenAddr)
	if err != nil {
		log.Fatalf("[udp] resolve: %v", err)
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("[udp] listen: %v", err)
	}
	if err := conn.SetReadBuffer(4 * 1024 * 1024); err != nil {
		log.Printf("[udp] SetReadBuffer: %v", err)
	}
	defer conn.Close()

	fmt.Println("════════════════════════════════════════════════")
	fmt.Println("  Traditional Backend  — http://localhost:8081")
	fmt.Println("  Stats               — http://localhost:8081/stats")
	fmt.Println("  UDP listening on    — 127.0.0.1:5001")
	fmt.Println("════════════════════════════════════════════════")

	buf := make([]byte, 65535)
	for {
		n, _, err := conn.ReadFromUDP(buf)
		if err != nil {
			log.Printf("[udp] read: %v", err)
			continue
		}
		if n < HeaderSize {
			continue
		}

		// Parse header
		frameID := binary.BigEndian.Uint32(buf[0:4])
		seq := int(binary.BigEndian.Uint16(buf[4:6]))
		total := int(binary.BigEndian.Uint16(buf[6:8]))
		ts := binary.BigEndian.Uint64(buf[8:16])

		payload := make([]byte, n-HeaderSize)
		copy(payload, buf[HeaderSize:n])

		// Track raw bytes received (real wire cost)
		m.recordUDPIn(n)

		full, frameTs, ok := reassembler.add(frameID, seq, total, payload, ts)
		if !ok {
			continue
		}

		go func(data []byte, edgeTs uint64) {
			// Validate JPEG
			if _, err := jpeg.Decode(bytes.NewReader(data)); err != nil {
				log.Printf("[process] bad JPEG: %v", err)
				return
			}

			m.recordLatency(edgeTs)

			select {
			case hub.broadcast <- data:
				m.recordWSOut(len(data))
			default:
				m.recordDrop()
			}
		}(full, frameTs)
	}
}

// ─────────────────────────────────────────────
//  Embedded frontend HTML
// ─────────────────────────────────────────────

const indexHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Traditional Stream</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --red:    #e63946;
    --amber:  #f4a261;
    --bg:     #0d0d0d;
    --panel:  #161616;
    --border: #2a2a2a;
    --text:   #c9c9c9;
    --dim:    #555;
    --mono:   'IBM Plex Mono', monospace;
    --sans:   'IBM Plex Sans', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px 40px;
    gap: 16px;
  }

  header { text-align: center; }
  header h1 {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.04em;
  }
  .badge {
    display: inline-block;
    margin-top: 6px;
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    padding: 3px 10px;
    border: 1px solid var(--red);
    color: var(--red);
    border-radius: 2px;
  }
  header p {
    margin-top: 8px;
    font-size: 0.75rem;
    color: var(--dim);
    font-weight: 300;
  }

  #wrap {
    position: relative;
    width: 100%;
    max-width: 800px;
    background: #000;
    border: 1px solid var(--border);
  }
  canvas { display: block; width: 100%; height: auto; }

  #hud {
    position: absolute;
    top: 10px; left: 10px;
    display: flex; flex-direction: column; gap: 5px;
    pointer-events: none;
  }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(13,13,13,0.8);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 3px 10px 3px 7px;
    font-family: var(--mono);
    font-size: 0.68rem;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dim); flex-shrink: 0; }
  .dot.live  { background: var(--red);   box-shadow: 0 0 6px var(--red); }
  .dot.warn  { background: var(--amber); box-shadow: 0 0 6px var(--amber); }

  #statsBar {
    width: 100%; max-width: 800px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }
  .stat {
    background: var(--panel);
    padding: 12px 14px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .stat-label {
    font-family: var(--mono);
    font-size: 0.55rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--dim);
  }
  .stat-value {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
  }
  .stat-value.bad { color: var(--red); }

  .compare-link {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--dim);
    text-align: center;
  }
  .compare-link a { color: #4cc9f0; text-decoration: none; }
  .compare-link a:hover { text-decoration: underline; }

  #foot {
    width: 100%; max-width: 800px;
    display: flex; justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--dim);
  }
  #connState.live { color: var(--red); }
  #connState.warn { color: var(--amber); }
</style>
</head>
<body>

<header>
  <h1>Traditional Full-Frame Stream</h1>
  <div class="badge">NO SEGMENTATION — FULL FRAME — 80% JPEG</div>
  <p>Every pixel sent every frame · No object detection · No bandwidth intelligence</p>
</header>

<div id="wrap">
  <canvas id="c"></canvas>
  <div id="hud">
    <div class="pill"><span class="dot" id="dot"></span><span id="connLabel">Connecting...</span></div>
    <div class="pill">FPS <span id="hudFps">—</span></div>
    <div class="pill">BW <span id="hudBw">—</span></div>
  </div>
</div>

<div id="statsBar">
  <div class="stat"><span class="stat-label">FPS In (UDP)</span><span class="stat-value" id="sFpsIn">—</span></div>
  <div class="stat"><span class="stat-label">FPS Out (WS)</span><span class="stat-value" id="sFpsOut">—</span></div>
  <div class="stat"><span class="stat-label">BW In (KB/s)</span><span class="stat-value bad" id="sBwIn">—</span></div>
  <div class="stat"><span class="stat-label">BW Out (KB/s)</span><span class="stat-value" id="sBwOut">—</span></div>
  <div class="stat"><span class="stat-label">Latency P50</span><span class="stat-value" id="sP50">—</span></div>
  <div class="stat"><span class="stat-label">Latency P95</span><span class="stat-value" id="sP95">—</span></div>
  <div class="stat"><span class="stat-label">Dropped</span><span class="stat-value" id="sDrop">—</span></div>
  <div class="stat"><span class="stat-label">Clients</span><span class="stat-value" id="sClients">—</span></div>
</div>

<div class="compare-link">
  Compare with SASP at <a href="http://localhost:8080" target="_blank">localhost:8080</a>
  — same camera, same FPS target, dramatically less bandwidth
</div>

<div id="foot">
  <span id="connState" class="warn">⬤ Connecting...</span>
  <span>Traditional Backend v1</span>
</div>

<script>
"use strict";

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

// ── Renderer ─────────────────────────────────────────────────────────────────
let pending = null, scheduled = false;

function scheduleRender(bm) {
  if (pending) pending.close();
  pending = bm;
  if (!scheduled) { scheduled = true; requestAnimationFrame(paint); }
}
function paint() {
  scheduled = false;
  if (!pending) return;
  const bm = pending; pending = null;
  if (canvas.width !== bm.width || canvas.height !== bm.height) {
    canvas.width = bm.width; canvas.height = bm.height;
  }
  ctx.drawImage(bm, 0, 0);
  bm.close();
}

// ── Client-side metrics ───────────────────────────────────────────────────────
let rxF = 0, rxB = 0, latSamp = [], lastT = performance.now();

function recordFrame(bytes, lat) {
  rxF++; rxB += bytes;
  if (latSamp.length >= 60) latSamp.shift();
  latSamp.push(lat);
}

setInterval(() => {
  const now = performance.now();
  const el  = (now - lastT) / 1000;
  const fps = rxF / el;
  const bw  = rxB / el / 1024;
  document.getElementById('hudFps').textContent = fps.toFixed(1);
  document.getElementById('hudBw').textContent  = bw >= 1024
    ? (bw/1024).toFixed(2)+' MB/s' : bw.toFixed(1)+' KB/s';
  rxF = 0; rxB = 0; lastT = now;
}, 1000);

// ── /stats polling ────────────────────────────────────────────────────────────
async function pollStats() {
  try {
    const r = await fetch('/stats');
    const j = await r.json();
    const d = j.data;
    document.getElementById('sFpsIn').textContent   = d.fps_in.toFixed(1);
    document.getElementById('sFpsOut').textContent  = d.fps_out.toFixed(1);
    document.getElementById('sBwIn').textContent    = d.bandwidth_in_kbs.toFixed(1);
    document.getElementById('sBwOut').textContent   = d.bandwidth_out_kbs.toFixed(1);
    document.getElementById('sP50').textContent     = d.latency_p50_ms.toFixed(1)+' ms';
    document.getElementById('sP95').textContent     = d.latency_p95_ms.toFixed(1)+' ms';
    document.getElementById('sDrop').textContent    = d.dropped_frames;
    document.getElementById('sClients').textContent = d.active_clients;
    document.getElementById('sDrop').classList.toggle('bad', d.dropped_frames > 0);
  } catch(_) {}
}
setInterval(pollStats, 1000);

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(s) {
  const map = {
    connecting: { dot:'warn',  label:'Connecting...',  cs:'warn' },
    live:       { dot:'live',  label:'Live',            cs:'live' },
    retry:      { dot:'warn',  label:'Reconnecting...', cs:'warn' },
    error:      { dot:'',      label:'Disconnected',    cs:''     },
  };
  const c = map[s] || map.error;
  document.getElementById('dot').className       = 'dot ' + c.dot;
  document.getElementById('connLabel').textContent = c.label;
  const cs = document.getElementById('connState');
  cs.className = c.cs;
  cs.textContent = s === 'live' ? '⬤ Stream active' : '⬤ ' + c.label;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
const WS = 'ws://' + location.host + '/ws';
let retryDelay = 500, ws;

function connect() {
  setStatus('connecting');
  ws = new WebSocket(WS);
  ws.binaryType = 'blob';
  ws.onopen  = () => { setStatus('live'); retryDelay = 500; };
  ws.onmessage = async (ev) => {
    const t0 = performance.now();
    try {
      const bm = await createImageBitmap(ev.data);
      recordFrame(ev.data.size, performance.now() - t0);
      scheduleRender(bm);
    } catch(_) {}
  };
  ws.onclose = () => {
    setStatus('retry');
    setTimeout(() => { retryDelay = Math.min(retryDelay*1.5, 8000); connect(); }, retryDelay);
  };
  ws.onerror = () => setStatus('error');
}
connect();
</script>
</body>
</html>`

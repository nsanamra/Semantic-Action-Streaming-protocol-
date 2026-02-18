package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"log"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// ─────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────

const (
	UDPAddr    = "127.0.0.1:5001" // traditional streamer sends here
	HTTPAddr   = ":8081"          // browser opens localhost:8081
	HeaderSize = 8                // FrameID(4) + SeqNum(2) + TotalParts(2)
	BufSize    = 12
)

// ─────────────────────────────────────────────
//  Stats — identical shape to SASP /stats JSON
//  so the comparison frontend can poll both the
//  same way and diff the numbers directly.
// ─────────────────────────────────────────────

type Stats struct {
	mu            sync.Mutex
	FramesIn      uint64
	FramesOut     uint64
	BytesIn       uint64 // raw UDP bytes received (true wire cost)
	BytesOut      uint64 // bytes pushed to WebSocket clients
	DroppedFrames uint64
	ActiveClients int32
	AvgFPS        float64
	lastTime      time.Time
	lastOut       uint64
}

func (s *Stats) udpIn(n int) {
	atomic.AddUint64(&s.FramesIn, 1)
	atomic.AddUint64(&s.BytesIn, uint64(n))
}
func (s *Stats) frameOut(n int) {
	atomic.AddUint64(&s.FramesOut, 1)
	atomic.AddUint64(&s.BytesOut, uint64(n))
}
func (s *Stats) dropped() { atomic.AddUint64(&s.DroppedFrames, 1) }

func (s *Stats) JSON() []byte {
	now := time.Now()
	out := atomic.LoadUint64(&s.FramesOut)
	s.mu.Lock()
	elapsed := now.Sub(s.lastTime).Seconds()
	if elapsed > 0 {
		s.AvgFPS = float64(out-s.lastOut) / elapsed
	}
	s.lastOut = out
	s.lastTime = now
	avgFPS := s.AvgFPS
	s.mu.Unlock()

	bin := atomic.LoadUint64(&s.BytesIn)
	bout := atomic.LoadUint64(&s.BytesOut)
	fps := avgFPS

	// bytes_in_per_sec = raw UDP bandwidth consumed (the real cost number)
	m := map[string]any{
		"mode":           "traditional",
		"frames_in":      atomic.LoadUint64(&s.FramesIn),
		"frames_out":     out,
		"bytes_in":       bin,
		"bytes_out":      bout,
		"dropped_frames": atomic.LoadUint64(&s.DroppedFrames),
		"active_clients": atomic.LoadInt32(&s.ActiveClients),
		"avg_fps":        fmt.Sprintf("%.1f", fps),
		"avg_persons":    "N/A", // traditional has no semantic awareness
		"jpeg_quality":   "80%",
		"encoding":       "full-frame JPEG",
	}
	b, _ := json.Marshal(m)
	return b
}

var stats = &Stats{lastTime: time.Now()}

// ─────────────────────────────────────────────
//  Reassembler  (minimal — no ChunkKey needed,
//  traditional only has one stream type)
// ─────────────────────────────────────────────

type FrameBuf struct {
	parts     [][]byte
	received  int
	total     int
	createdAt time.Time
}

type Reassembler struct {
	mu     sync.Mutex
	frames map[uint32]*FrameBuf
}

func NewReassembler() *Reassembler {
	r := &Reassembler{frames: make(map[uint32]*FrameBuf)}
	go r.gcLoop()
	return r
}

func (r *Reassembler) Add(id uint32, seq, total int, data []byte) ([]byte, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	fb, ok := r.frames[id]
	if !ok {
		fb = &FrameBuf{parts: make([][]byte, total), total: total, createdAt: time.Now()}
		r.frames[id] = fb
	}
	if seq >= 0 && seq < len(fb.parts) && fb.parts[seq] == nil {
		fb.parts[seq] = data
		fb.received++
	}
	if fb.received == fb.total {
		full := bytes.Join(fb.parts, nil)
		delete(r.frames, id)
		return full, true
	}
	return nil, false
}

func (r *Reassembler) gcLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		r.mu.Lock()
		for id, fb := range r.frames {
			if time.Since(fb.createdAt) > 2*time.Second {
				delete(r.frames, id)
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

func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*client]struct{}),
		register:   make(chan *client, 8),
		unregister: make(chan *client, 8),
		broadcast:  make(chan []byte, BufSize),
	}
}

func (h *Hub) run() {
	for {
		select {
		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = struct{}{}
			h.mu.Unlock()
			atomic.AddInt32(&stats.ActiveClients, 1)
			log.Printf("[hub] +client  total=%d", atomic.LoadInt32(&stats.ActiveClients))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				c.closeOnce.Do(func() { close(c.send) })
			}
			h.mu.Unlock()
			atomic.AddInt32(&stats.ActiveClients, -1)
			log.Printf("[hub] -client  total=%d", atomic.LoadInt32(&stats.ActiveClients))

		case frame := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- frame:
				default:
					stats.dropped()
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
	c := &client{conn: conn, send: make(chan []byte, 6)}
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
	hub := NewHub()
	reassembler := NewReassembler()

	go hub.run()

	// Serve the traditional frontend + WebSocket
	http.Handle("/", http.FileServer(http.Dir("./public_trad")))
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		serveWS(hub, w, r)
	})
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Write(stats.JSON())
	})

	go func() {
		log.Printf("[http] Traditional backend → http://localhost%s", HTTPAddr)
		if err := http.ListenAndServe(HTTPAddr, nil); err != nil {
			log.Fatalf("[http] fatal: %v", err)
		}
	}()

	// UDP listener
	addr, err := net.ResolveUDPAddr("udp", UDPAddr)
	if err != nil {
		log.Fatalf("[udp] resolve: %v", err)
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("[udp] listen: %v", err)
	}
	defer conn.Close()

	fmt.Println("══════════════════════════════════════════════════")
	fmt.Println("  Traditional Backend  —  http://localhost:8081")
	fmt.Println("  Stats               —  http://localhost:8081/stats")
	fmt.Println("  UDP listening on    —  127.0.0.1:5001")
	fmt.Println("══════════════════════════════════════════════════")

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

		stats.udpIn(n)

		frameID := binary.BigEndian.Uint32(buf[0:4])
		seq := int(binary.BigEndian.Uint16(buf[4:6]))
		total := int(binary.BigEndian.Uint16(buf[6:8]))

		payload := make([]byte, n-HeaderSize)
		copy(payload, buf[HeaderSize:n])

		if fullFrame, ok := reassembler.Add(frameID, seq, total, payload); ok {
			go func(data []byte) {
				// Validate it's a real JPEG before broadcasting
				if _, err := jpeg.Decode(bytes.NewReader(data)); err != nil {
					log.Printf("[process] bad JPEG: %v", err)
					return
				}
				select {
				case hub.broadcast <- data:
					stats.frameOut(len(data))
				default:
					stats.dropped()
				}
			}(fullFrame)
		}
	}
}

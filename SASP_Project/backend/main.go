package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// ─────────────────────────────────────────────
//  Protocol constants
// ─────────────────────────────────────────────

const (
	HeaderSize = 28

	TypeBackground byte = 0
	TypeROI        byte = 1

	// Fallback: if Python sends roi_count=0 in a BG packet that claims to
	// have objects, we fall back to the collect-window approach.
	ROIFallbackWindow = 8 * time.Millisecond

	// Broadcast buffer — frames dropped for slow clients, never block pipeline
	BroadcastBufferSize = 32
)

// ─────────────────────────────────────────────
//  Stats  (served at /stats as JSON)
// ─────────────────────────────────────────────

type Stats struct {
	mu            sync.Mutex
	FramesIn      uint64
	FramesOut     uint64
	BytesOut      uint64
	DroppedFrames uint64
	ActiveClients int32
	PersonsTotal  uint64 // cumulative persons stitched
	AvgFPS        float64
	AvgPersons    float64
	lastTime      time.Time
	lastOut       uint64
	lastPersons   uint64
}

func (s *Stats) frameIn() { atomic.AddUint64(&s.FramesIn, 1) }
func (s *Stats) frameOut(n int) {
	atomic.AddUint64(&s.FramesOut, 1)
	atomic.AddUint64(&s.BytesOut, uint64(n))
}
func (s *Stats) dropped()         { atomic.AddUint64(&s.DroppedFrames, 1) }
func (s *Stats) addPersons(n int) { atomic.AddUint64(&s.PersonsTotal, uint64(n)) }

func (s *Stats) JSON() []byte {
	now := time.Now()
	out := atomic.LoadUint64(&s.FramesOut)
	ppl := atomic.LoadUint64(&s.PersonsTotal)
	s.mu.Lock()
	elapsed := now.Sub(s.lastTime).Seconds()
	if elapsed > 0 {
		s.AvgFPS = float64(out-s.lastOut) / elapsed
		s.AvgPersons = float64(ppl-s.lastPersons) / elapsed
		// Normalise persons-per-second → persons-per-frame
		if s.AvgFPS > 0 {
			s.AvgPersons = s.AvgPersons / s.AvgFPS
		}
	}
	s.lastOut = out
	s.lastPersons = ppl
	s.lastTime = now
	avgFPS := s.AvgFPS
	avgPersons := s.AvgPersons
	s.mu.Unlock()

	m := map[string]any{
		"frames_in":      atomic.LoadUint64(&s.FramesIn),
		"frames_out":     out,
		"bytes_out":      atomic.LoadUint64(&s.BytesOut),
		"dropped_frames": atomic.LoadUint64(&s.DroppedFrames),
		"active_clients": atomic.LoadInt32(&s.ActiveClients),
		"avg_fps":        fmt.Sprintf("%.1f", avgFPS),
		"avg_persons":    fmt.Sprintf("%.1f", avgPersons),
	}
	b, _ := json.Marshal(m)
	return b
}

var globalStats = &Stats{lastTime: time.Now()}

// ─────────────────────────────────────────────
//  FrameSyncer
//
//  Two flush strategies, chosen per-frame:
//
//  A) Exact-count flush (preferred):
//     Python embeds roi_count (e.g. 2) in the BG header.
//     Go waits until exactly 2 ROI tiles arrive, then flushes
//     immediately — zero latency tax.
//
//  B) Fallback window (safety net):
//     If roi_count arrived as 0 but fClass says objects exist
//     (e.g. older Python client), we use a 15ms collect window
//     exactly as before.
// ─────────────────────────────────────────────

type pendingFrame struct {
	canvas      *image.RGBA
	arrivedAt   time.Time
	firstROI    time.Time
	roiExpected int // 0 = unknown (use window), N = exact count
	roiReceived int
	flushed     bool
	timer       *time.Timer // non-nil only in window-mode
}

type FrameSyncer struct {
	mu      sync.Mutex
	pending map[uint32]*pendingFrame
	hub     *Hub
}

func NewFrameSyncer(hub *Hub) *FrameSyncer {
	fs := &FrameSyncer{pending: make(map[uint32]*pendingFrame), hub: hub}
	go fs.gcLoop()
	return fs
}

// StoreBG stores the decoded background for frameID.
// roiCount is the exact number of ROI tiles Python will send (0 = none).
func (fs *FrameSyncer) StoreBG(frameID uint32, bg image.Image, roiCount int) (image.Image, bool) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if roiCount == 0 {
		// Fast path: no ROIs coming — emit BG immediately
		delete(fs.pending, frameID)
		return bg, true
	}

	bounds := bg.Bounds()
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, bg, image.Point{}, draw.Src)

	fs.pending[frameID] = &pendingFrame{
		canvas:      canvas,
		arrivedAt:   time.Now(),
		roiExpected: roiCount,
	}
	return nil, false
}

// AddROI paints one ROI tile onto the canvas and flushes when done.
func (fs *FrameSyncer) AddROI(frameID uint32, roi image.Image, x, y int) {
	fs.mu.Lock()

	pf, ok := fs.pending[frameID]
	if !ok || pf.flushed {
		fs.mu.Unlock()
		return
	}

	// Paint
	bounds := pf.canvas.Bounds()
	dstRect := image.Rect(x, y, x+roi.Bounds().Dx(), y+roi.Bounds().Dy()).Intersect(bounds)
	if !dstRect.Empty() {
		draw.Draw(pf.canvas, dstRect, roi, image.Point{}, draw.Over)
	}
	pf.roiReceived++

	isFirst := pf.firstROI.IsZero()
	if isFirst {
		pf.firstROI = time.Now()
	}

	// ── Flush strategy decision ───────────────────────────────────────────
	if pf.roiExpected > 0 {
		// Strategy A: exact count known — flush the instant we have them all
		if pf.roiReceived >= pf.roiExpected {
			pf.flushed = true
			canvas := pf.canvas
			received := pf.roiReceived
			delete(fs.pending, frameID)
			fs.mu.Unlock()
			log.Printf("[sync] frame %d flushed (exact: %d ROIs)", frameID, received)
			encodeAndBroadcast(canvas, fs.hub)
			return
		}
		fs.mu.Unlock()
		return
	}

	// Strategy B: count unknown — start collect window on first ROI
	if isFirst {
		pf.timer = time.AfterFunc(ROIFallbackWindow, func() {
			fs.flushFrame(frameID)
		})
	}
	fs.mu.Unlock()
}

func (fs *FrameSyncer) flushFrame(frameID uint32) {
	fs.mu.Lock()
	pf, ok := fs.pending[frameID]
	if !ok || pf.flushed {
		fs.mu.Unlock()
		return
	}
	pf.flushed = true
	if pf.timer != nil {
		pf.timer.Stop()
	}
	canvas := pf.canvas
	received := pf.roiReceived
	delete(fs.pending, frameID)
	fs.mu.Unlock()

	log.Printf("[sync] frame %d flushed (window: %d ROIs)", frameID, received)
	encodeAndBroadcast(canvas, fs.hub)
}

func (fs *FrameSyncer) gcLoop() {
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		fs.mu.Lock()
		var stale []uint32
		for id, pf := range fs.pending {
			if !pf.flushed && time.Since(pf.arrivedAt) > 40*time.Millisecond {
				stale = append(stale, id)
			}
		}
		fs.mu.Unlock()
		for _, id := range stale {
			log.Printf("[sync] evicting stale frame %d", id)
			fs.flushFrame(id)
		}
	}
}

// ─────────────────────────────────────────────
//  Reassembler
// ─────────────────────────────────────────────

type ChunkKey struct {
	ID   uint32
	Type byte
}

type FrameBuffer struct {
	Parts     [][]byte
	Received  int
	Total     int
	X, Y      int
	ROICount  int
	ROIIndex  int
	CreatedAt time.Time
}

type Reassembler struct {
	mu     sync.Mutex
	frames map[ChunkKey]*FrameBuffer
}

func NewReassembler() *Reassembler {
	r := &Reassembler{frames: make(map[ChunkKey]*FrameBuffer)}
	go r.gcLoop()
	return r
}

func (r *Reassembler) AddPart(
	id, roiCount, roiIndex uint32,
	fType byte,
	seq, total int,
	data []byte,
	x, y int,
) (payload []byte, ox, oy, oRoiCount int, complete bool) {

	r.mu.Lock()
	defer r.mu.Unlock()

	key := ChunkKey{ID: id, Type: fType}
	fb, ok := r.frames[key]
	if !ok {
		fb = &FrameBuffer{
			Parts:     make([][]byte, total),
			Total:     total,
			ROICount:  int(roiCount),
			ROIIndex:  int(roiIndex),
			CreatedAt: time.Now(),
		}
		r.frames[key] = fb
	}
	if seq >= 0 && seq < len(fb.Parts) && fb.Parts[seq] == nil {
		fb.Parts[seq] = data
		fb.Received++
		fb.X = x
		fb.Y = y
	}
	if fb.Received == fb.Total {
		full := bytes.Join(fb.Parts, nil)
		delete(r.frames, key)
		return full, fb.X, fb.Y, fb.ROICount, true
	}
	return nil, 0, 0, 0, false
}

func (r *Reassembler) gcLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		r.mu.Lock()
		for k, fb := range r.frames {
			if time.Since(fb.CreatedAt) > 2*time.Second {
				delete(r.frames, k)
				log.Printf("[reassembler] evicted stale key=%+v", k)
			}
		}
		r.mu.Unlock()
	}
}

// ─────────────────────────────────────────────
//  WebSocket Hub
// ─────────────────────────────────────────────

type wsClient struct {
	conn      *websocket.Conn
	send      chan []byte
	closeOnce sync.Once
}

func (c *wsClient) writePump() {
	defer c.conn.Close()
	for frame := range c.send {
		if err := c.conn.WriteMessage(websocket.BinaryMessage, frame); err != nil {
			return
		}
	}
}

type Hub struct {
	mu         sync.RWMutex
	clients    map[*wsClient]struct{}
	register   chan *wsClient
	unregister chan *wsClient
	broadcast  chan []byte
}

func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*wsClient]struct{}),
		register:   make(chan *wsClient, 8),
		unregister: make(chan *wsClient, 8),
		broadcast:  make(chan []byte, BroadcastBufferSize),
	}
}

func (h *Hub) run() {
	for {
		select {
		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = struct{}{}
			h.mu.Unlock()
			atomic.AddInt32(&globalStats.ActiveClients, 1)
			log.Printf("[hub] client connected (total=%d)", atomic.LoadInt32(&globalStats.ActiveClients))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				c.closeOnce.Do(func() { close(c.send) })
			}
			h.mu.Unlock()
			atomic.AddInt32(&globalStats.ActiveClients, -1)
			log.Printf("[hub] client disconnected (total=%d)", atomic.LoadInt32(&globalStats.ActiveClients))

		case frame := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- frame:
				default:
					globalStats.dropped()
					log.Printf("[hub] slow client — frame dropped")
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

func (h *Hub) serveWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[ws] upgrade: %v", err)
		return
	}
	c := &wsClient{conn: conn, send: make(chan []byte, 6)}
	h.register <- c
	go c.writePump()
	go func() {
		defer func() { h.unregister <- c }()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}()
}

// ─────────────────────────────────────────────
//  Frame processing
// ─────────────────────────────────────────────

func processFrame(
	data []byte,
	fType byte,
	frameID uint32,
	roiCount int,
	x, y int,
	hub *Hub,
	syncer *FrameSyncer,
) {
	globalStats.frameIn()

	var img image.Image
	var err error

	switch fType {
	case TypeBackground:
		img, err = jpeg.Decode(bytes.NewReader(data))
	case TypeROI:
		img, err = png.Decode(bytes.NewReader(data))
	default:
		return
	}
	if err != nil {
		log.Printf("[process] decode error fType=%d frameID=%d: %v", fType, frameID, err)
		return
	}

	switch fType {
	case TypeBackground:
		result, ready := syncer.StoreBG(frameID, img, roiCount)
		if ready {
			encodeAndBroadcast(result, hub)
		}
	case TypeROI:
		syncer.AddROI(frameID, img, x, y)
		if roiCount > 0 {
			globalStats.addPersons(roiCount)
		}
	}
}

func encodeAndBroadcast(img image.Image, hub *Hub) {
	var out bytes.Buffer
	if err := jpeg.Encode(&out, img, &jpeg.Options{Quality: 82}); err != nil {
		log.Printf("[encode] %v", err)
		return
	}
	b := out.Bytes()
	globalStats.frameOut(len(b))

	select {
	case hub.broadcast <- b:
	default:
		globalStats.dropped()
		log.Printf("[broadcast] channel full — frame dropped")
	}
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

func main() {
	hub := NewHub()
	reassembler := NewReassembler()
	syncer := NewFrameSyncer(hub)

	go hub.run()

	// ── HTTP ─────────────────────────────────────────────────────────────────
	http.Handle("/", http.FileServer(http.Dir("./public")))
	http.HandleFunc("/ws", hub.serveWS)

	// Stats endpoint — polled by the frontend every second
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Write(globalStats.JSON())
	})

	go func() {
		log.Println("[http] → http://localhost:8080")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("[http] fatal: %v", err)
		}
	}()

	// ── UDP ──────────────────────────────────────────────────────────────────
	addr, err := net.ResolveUDPAddr("udp", "127.0.0.1:5000")
	if err != nil {
		log.Fatalf("[udp] resolve: %v", err)
	}
	udpConn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("[udp] listen: %v", err)
	}
	if err := udpConn.SetReadBuffer(4 * 1024 * 1024); err != nil {
		log.Printf("[udp] SetReadBuffer: %v", err)
	}
	defer udpConn.Close()

	fmt.Println("════════════════════════════════════════════")
	fmt.Println("  SASP Backend v3  —  http://localhost:8080 ")
	fmt.Println("  Stats           —  http://localhost:8080/stats")
	fmt.Println("════════════════════════════════════════════")

	buf := make([]byte, 65535)
	for {
		n, _, err := udpConn.ReadFromUDP(buf)
		if err != nil {
			log.Printf("[udp] read: %v", err)
			continue
		}
		if n < HeaderSize {
			continue
		}

		// ── Parse 28-byte header ──────────────────────────────────────────────
		// 0-3   Magic
		// 4     Version
		// 5     Type
		// 6-9   FrameID
		// 10-11 SeqNum
		// 12-13 TotalParts
		// 14-21 Timestamp
		// 22    ROICount  (total tiles expected for this frame)
		// 23    ROIIndex  (which tile this packet is)
		// 24-25 X
		// 26-27 Y
		fType := buf[5]
		fID := binary.BigEndian.Uint32(buf[6:10])
		seq := binary.BigEndian.Uint16(buf[10:12])
		total := binary.BigEndian.Uint16(buf[12:14])
		roiCount := int(buf[22])
		// roiIndex := int(buf[23])  // available if needed for debugging
		xPos := int(binary.BigEndian.Uint16(buf[24:26]))
		yPos := int(binary.BigEndian.Uint16(buf[26:28]))

		payload := make([]byte, n-HeaderSize)
		copy(payload, buf[HeaderSize:n])

		full, x, y, rc, ok := reassembler.AddPart(
			fID, uint32(roiCount), uint32(buf[23]),
			fType, int(seq), int(total),
			payload, xPos, yPos,
		)
		if ok {
			go processFrame(full, fType, fID, rc, x, y, hub, syncer)
		}
	}
}

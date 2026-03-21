package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png" // register PNG decoder for image.Decode
	"log"
	"math"
	"net"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	_ "golang.org/x/image/webp" // register WebP decoder for image.Decode
)

// ─────────────────────────────────────────────
//  Protocol constants
// ─────────────────────────────────────────────

const (
	HeaderSize = 28

	TypeBackground byte = 0
	TypeROI        byte = 1

	// Fallback: if exact ROI count is unknown, collect window before flushing.
	// 8ms ≈ ¼ of a 30 FPS frame — tight but sufficient for LAN.
	ROIFallbackWindow = 8 * time.Millisecond

	// Broadcast buffer — frames dropped for slow clients, never block pipeline
	BroadcastBufferSize = 32
)

// ─────────────────────────────────────────────
//  MetricsEngine — robust tumbling-window metrics
//
//  Instead of computing FPS relative to whenever
//  the frontend happens to poll, we snapshot every
//  1 second atomically.  All values are pre-computed
//  and served instantly via /api/metrics.
// ─────────────────────────────────────────────

type MetricsEngine struct {
	// ── Atomic counters (incremented on hot path) ──
	framesIn     uint64
	framesOut    uint64
	bytesIn      uint64
	bytesOut     uint64
	droppedOut   uint64
	personsTotal uint64

	// ── Latency ring buffer (lock-guarded) ──
	latMu      sync.Mutex
	latSamples []float64 // milliseconds, last 300 samples (~10s)

	// ── Snapshot (read by /api/metrics) ──
	snapMu   sync.RWMutex
	snapshot MetricsSnapshot

	// ── Client counter ──
	activeClients int32

	startTime time.Time
}

type MetricsSnapshot struct {
	FPSIn           float64 `json:"fps_in"`
	FPSOut          float64 `json:"fps_out"`
	BandwidthInKBs  float64 `json:"bandwidth_in_kbps"`
	BandwidthOutKBs float64 `json:"bandwidth_out_kbps"`
	LatencyP50Ms    float64 `json:"latency_p50_ms"`
	LatencyP95Ms    float64 `json:"latency_p95_ms"`
	LatencyP99Ms    float64 `json:"latency_p99_ms"`
	PersonsPerFrame float64 `json:"persons_per_frame"`
	DroppedFrames   uint64  `json:"dropped_frames"`
	ActiveClients   int32   `json:"active_clients"`
	UptimeSeconds   float64 `json:"uptime_seconds"`
}

func NewMetricsEngine() *MetricsEngine {
	m := &MetricsEngine{
		latSamples: make([]float64, 0, 300),
		startTime:  time.Now(),
	}
	go m.snapshotLoop()
	return m
}

// ── Hot-path recorders (lock-free atomics) ──

func (m *MetricsEngine) RecordFrameIn(bytesReceived int) {
	atomic.AddUint64(&m.framesIn, 1)
	atomic.AddUint64(&m.bytesIn, uint64(bytesReceived))
}

func (m *MetricsEngine) RecordFrameOut(bytesSent int) {
	atomic.AddUint64(&m.framesOut, 1)
	atomic.AddUint64(&m.bytesOut, uint64(bytesSent))
}

func (m *MetricsEngine) RecordDrop() {
	atomic.AddUint64(&m.droppedOut, 1)
}

func (m *MetricsEngine) RecordPersons(n int) {
	atomic.AddUint64(&m.personsTotal, uint64(n))
}

func (m *MetricsEngine) RecordLatency(edgeTimestampNs uint64) {
	nowNs := uint64(time.Now().UnixNano())
	if edgeTimestampNs > nowNs || edgeTimestampNs == 0 {
		return // clock skew or missing timestamp
	}
	latMs := float64(nowNs-edgeTimestampNs) / 1e6
	if latMs > 5000 {
		return // discard absurd values
	}

	m.latMu.Lock()
	if len(m.latSamples) >= 300 {
		// shift out oldest
		m.latSamples = m.latSamples[1:]
	}
	m.latSamples = append(m.latSamples, latMs)
	m.latMu.Unlock()
}

func (m *MetricsEngine) ClientConnect()    { atomic.AddInt32(&m.activeClients, 1) }
func (m *MetricsEngine) ClientDisconnect() { atomic.AddInt32(&m.activeClients, -1) }

// ── Snapshot loop (runs every 1 second) ──

func (m *MetricsEngine) snapshotLoop() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var prevIn, prevOut, prevBytes, prevBytesOut, prevPersons uint64

	for range ticker.C {
		curIn := atomic.LoadUint64(&m.framesIn)
		curOut := atomic.LoadUint64(&m.framesOut)
		curBytesIn := atomic.LoadUint64(&m.bytesIn)
		curBytesOut := atomic.LoadUint64(&m.bytesOut)
		curPersons := atomic.LoadUint64(&m.personsTotal)

		deltaIn := curIn - prevIn
		deltaOut := curOut - prevOut
		deltaBytesIn := curBytesIn - prevBytes
		deltaBytesOut := curBytesOut - prevBytesOut
		deltaPersons := curPersons - prevPersons

		prevIn = curIn
		prevOut = curOut
		prevBytes = curBytesIn
		prevBytesOut = curBytesOut
		prevPersons = curPersons

		var personsPerFrame float64
		if deltaOut > 0 {
			personsPerFrame = float64(deltaPersons) / float64(deltaOut)
		}

		// Compute latency percentiles
		p50, p95, p99 := m.computePercentiles()

		snap := MetricsSnapshot{
			FPSIn:           float64(deltaIn),
			FPSOut:          float64(deltaOut),
			BandwidthInKBs:  float64(deltaBytesIn) / 1024.0,
			BandwidthOutKBs: float64(deltaBytesOut) / 1024.0,
			LatencyP50Ms:    p50,
			LatencyP95Ms:    p95,
			LatencyP99Ms:    p99,
			PersonsPerFrame: personsPerFrame,
			DroppedFrames:   atomic.LoadUint64(&m.droppedOut),
			ActiveClients:   atomic.LoadInt32(&m.activeClients),
			UptimeSeconds:   time.Since(m.startTime).Seconds(),
		}

		m.snapMu.Lock()
		m.snapshot = snap
		m.snapMu.Unlock()
	}
}

func (m *MetricsEngine) computePercentiles() (p50, p95, p99 float64) {
	m.latMu.Lock()
	n := len(m.latSamples)
	if n == 0 {
		m.latMu.Unlock()
		return 0, 0, 0
	}
	sorted := make([]float64, n)
	copy(sorted, m.latSamples)
	m.latMu.Unlock()

	sort.Float64s(sorted)

	percentile := func(p float64) float64 {
		idx := int(math.Ceil(p/100.0*float64(n))) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= n {
			idx = n - 1
		}
		return math.Round(sorted[idx]*100) / 100
	}

	return percentile(50), percentile(95), percentile(99)
}

func (m *MetricsEngine) GetSnapshot() MetricsSnapshot {
	m.snapMu.RLock()
	defer m.snapMu.RUnlock()
	return m.snapshot
}

func (m *MetricsEngine) JSON() []byte {
	snap := m.GetSnapshot()
	b, _ := json.Marshal(snap)
	return b
}

var metrics = NewMetricsEngine()

// ─────────────────────────────────────────────
//  sync.Pool for per-packet byte slices
//  Eliminates ~150-450 allocs/sec on the hot path
// ─────────────────────────────────────────────

var packetPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 0, 2048) // typical packet size
		return &b
	},
}

func getPooledSlice(size int) []byte {
	bp := packetPool.Get().(*[]byte)
	b := *bp
	if cap(b) >= size {
		return b[:size]
	}
	// Rare: packet larger than pool capacity — allocate fresh
	return make([]byte, size)
}

func putPooledSlice(b []byte) {
	b = b[:0]
	packetPool.Put(&b)
}

// ─────────────────────────────────────────────
//  FrameSyncer
//
//  Flush rule: a frame is emitted ONLY when BOTH:
//    1. The background JPEG has been received, AND
//    2. All expected ROI tiles have arrived (or timeout).
//
//  This prevents black-frame strobes when ROI tiles
//  arrive before BG (which is common with ROI-first
//  send order on the edge).
//
//  Two count strategies:
//    A) Exact-count (preferred): roi_count from header.
//    B) Fallback window: 8ms collect after first ROI.
// ─────────────────────────────────────────────

type pendingFrame struct {
	canvas      *image.RGBA
	arrivedAt   time.Time
	firstROI    time.Time
	bgReceived  bool // true once StoreBG has been called
	roiExpected int  // 0 = unknown (use window), N = exact count
	roiReceived int
	roisDone    bool // true when all expected ROIs are in (or window expired)
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

// tryFlush checks if both BG and all ROIs are present; if so, flushes.
// MUST be called with fs.mu held. Returns true if flushed.
func (fs *FrameSyncer) tryFlush(frameID uint32, pf *pendingFrame) bool {
	if pf.flushed || !pf.bgReceived || !pf.roisDone {
		return false
	}
	pf.flushed = true
	if pf.timer != nil {
		pf.timer.Stop()
	}
	canvas := pf.canvas
	received := pf.roiReceived
	delete(fs.pending, frameID)
	fs.mu.Unlock()
	log.Printf("[sync] frame %d flushed (bg+%d ROIs)", frameID, received)
	encodeAndBroadcast(canvas, fs.hub)
	return true // caller must NOT unlock again
}

// StoreBG stores the decoded background for frameID.
func (fs *FrameSyncer) StoreBG(frameID uint32, bg image.Image, roiCount int) (image.Image, bool) {
	fs.mu.Lock()

	pf, exists := fs.pending[frameID]

	if roiCount == 0 && !exists {
		// Fast path: no ROIs coming, BG-only frame — emit immediately
		fs.mu.Unlock()
		return bg, true
	}

	if !exists {
		// BG arrived first (before any ROI) — create pending entry
		bounds := bg.Bounds()
		canvas := image.NewRGBA(bounds)
		draw.Draw(canvas, bounds, bg, image.Point{}, draw.Src)
		pf = &pendingFrame{
			canvas:      canvas,
			arrivedAt:   time.Now(),
			bgReceived:  true,
			roiExpected: roiCount,
		}
		fs.pending[frameID] = pf
		fs.mu.Unlock()
		return nil, false
	}

	// BG arrived AFTER some/all ROIs — paint BG underneath ROI composites
	bounds := bg.Bounds()
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, bg, image.Point{}, draw.Src)
	// Re-draw any ROIs that were already composited on top
	draw.Draw(canvas, pf.canvas.Bounds(), pf.canvas, image.Point{}, draw.Over)
	pf.canvas = canvas
	pf.bgReceived = true
	if pf.roiExpected == 0 {
		pf.roiExpected = roiCount
	}

	// Check if ROIs are also done — if so, flush now
	if pf.roiExpected > 0 && pf.roiReceived >= pf.roiExpected {
		pf.roisDone = true
	}
	if fs.tryFlush(frameID, pf) {
		return nil, false // tryFlush already unlocked
	}
	fs.mu.Unlock()
	return nil, false
}

// AddROI paints one ROI tile onto the canvas.
func (fs *FrameSyncer) AddROI(frameID uint32, roi image.Image, x, y int, roiCount int) {
	fs.mu.Lock()

	pf, ok := fs.pending[frameID]
	if !ok {
		// ROI arrived before BG — create placeholder canvas.
		// StoreBG will paint the real background underneath later.
		pf = &pendingFrame{
			canvas:      image.NewRGBA(image.Rect(0, 0, 640, 480)),
			arrivedAt:   time.Now(),
			roiExpected: roiCount,
		}
		fs.pending[frameID] = pf
	}
	if pf.flushed {
		fs.mu.Unlock()
		return
	}

	// Paint the ROI tile
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

	if pf.roiExpected == 0 && roiCount > 0 {
		pf.roiExpected = roiCount
	}

	// ── Check if all ROIs are in ──────────────────────────────────────────
	if pf.roiExpected > 0 && pf.roiReceived >= pf.roiExpected {
		pf.roisDone = true
		// Try to flush (only succeeds if BG is also received)
		if fs.tryFlush(frameID, pf) {
			return // tryFlush already unlocked
		}
		fs.mu.Unlock()
		return
	}

	// Strategy B: count unknown — start collect window on first ROI
	if pf.roiExpected == 0 && isFirst {
		pf.timer = time.AfterFunc(ROIFallbackWindow, func() {
			fs.mu.Lock()
			pf2, ok := fs.pending[frameID]
			if !ok || pf2.flushed {
				fs.mu.Unlock()
				return
			}
			pf2.roisDone = true // window expired, treat ROIs as complete
			if fs.tryFlush(frameID, pf2) {
				return
			}
			fs.mu.Unlock()
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

	log.Printf("[sync] frame %d force-flushed (gc: %d ROIs, bg=%v)", frameID, received, pf.bgReceived)
	encodeAndBroadcast(canvas, fs.hub)
}

func (fs *FrameSyncer) gcLoop() {
	// 200ms is plenty — exact-count flush handles 99%+ of frames instantly.
	// This is only a safety net for truly orphaned frames.
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		fs.mu.Lock()
		var stale []uint32
		for id, pf := range fs.pending {
			if !pf.flushed && time.Since(pf.arrivedAt) > 120*time.Millisecond {
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
	Timestamp uint64 // edge timestamp (ns) from first chunk
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
	timestamp uint64,
) (payload []byte, ox, oy, oRoiCount int, oTimestamp uint64, complete bool) {

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
			Timestamp: timestamp,
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
		ts := fb.Timestamp
		delete(r.frames, key)
		return full, fb.X, fb.Y, fb.ROICount, ts, true
	}
	return nil, 0, 0, 0, 0, false
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
			metrics.ClientConnect()
			log.Printf("[hub] client connected (total=%d)", atomic.LoadInt32(&metrics.activeClients))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				c.closeOnce.Do(func() { close(c.send) })
			}
			h.mu.Unlock()
			metrics.ClientDisconnect()
			log.Printf("[hub] client disconnected (total=%d)", atomic.LoadInt32(&metrics.activeClients))

		case frame := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- frame:
				default:
					metrics.RecordDrop()
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
	c := &wsClient{conn: conn, send: make(chan []byte, 24)} // bigger buffer for tab-switch tolerance
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
//  SmoothBuffer — jitter buffer + last-good-frame cache
//
//  Instead of broadcasting immediately (which causes
//  timing jitter when YOLO takes variable time to run),
//  frames are queued into a ring buffer and played back
//  at a steady 30fps by a ticker goroutine.
//
//  Last-good-frame: when a BG-only frame arrives but
//  recent frames had ROI composites, the BG-only frame
//  is replaced with the last composited frame. This
//  prevents the "momentary blur" flash when YOLO
//  temporarily loses a detection.
// ─────────────────────────────────────────────

const (
	SmoothBufferSize = 30 // ~1 second at 30fps
	PlaybackFPS      = 30
)

type bufferedFrame struct {
	data    []byte
	hasROIs bool
}

type SmoothBuffer struct {
	mu            sync.Mutex
	ring          []bufferedFrame
	writeIdx      int
	readIdx       int
	count         int
	lastGoodFrame []byte // last frame that had ROI compositing
	hub           *Hub
}

func NewSmoothBuffer(hub *Hub) *SmoothBuffer {
	sb := &SmoothBuffer{
		ring: make([]bufferedFrame, SmoothBufferSize),
		hub:  hub,
	}
	go sb.playbackLoop()
	return sb
}

// Push adds an encoded JPEG frame to the buffer.
// hasROIs indicates whether this frame had ROI compositing (not BG-only).
func (sb *SmoothBuffer) Push(data []byte, hasROIs bool) {
	sb.mu.Lock()
	defer sb.mu.Unlock()

	if hasROIs {
		sb.lastGoodFrame = data
	} else if sb.lastGoodFrame != nil {
		// BG-only frame but we have a recent composited frame — use that instead
		// This prevents the "momentary blur" when YOLO loses detection for 1-2 frames
		data = sb.lastGoodFrame
	}

	if sb.count == SmoothBufferSize {
		// Buffer full — drop oldest (advance read pointer)
		sb.readIdx = (sb.readIdx + 1) % SmoothBufferSize
		sb.count--
		metrics.RecordDrop()
	}

	sb.ring[sb.writeIdx] = bufferedFrame{data: data, hasROIs: hasROIs}
	sb.writeIdx = (sb.writeIdx + 1) % SmoothBufferSize
	sb.count++
}

// playbackLoop reads from the buffer at a steady 30fps and broadcasts.
func (sb *SmoothBuffer) playbackLoop() {
	ticker := time.NewTicker(time.Second / PlaybackFPS)
	defer ticker.Stop()

	for range ticker.C {
		sb.mu.Lock()
		if sb.count == 0 {
			// Buffer empty — if we have a last-good-frame, repeat it (prevents black/freeze)
			if sb.lastGoodFrame != nil {
				frame := sb.lastGoodFrame
				sb.mu.Unlock()
				sb.broadcast(frame)
			} else {
				sb.mu.Unlock()
			}
			continue
		}

		frame := sb.ring[sb.readIdx]
		sb.readIdx = (sb.readIdx + 1) % SmoothBufferSize
		sb.count--
		sb.mu.Unlock()

		sb.broadcast(frame.data)
	}
}

func (sb *SmoothBuffer) broadcast(data []byte) {
	metrics.RecordFrameOut(len(data))
	select {
	case sb.hub.broadcast <- data:
	default:
		metrics.RecordDrop()
	}
}

// ─────────────────────────────────────────────
//  Frame processing
// ─────────────────────────────────────────────

// Global smooth buffer — initialized in main()
var smoothBuffer *SmoothBuffer

func processFrame(
	data []byte,
	fType byte,
	frameID uint32,
	roiCount int,
	x, y int,
	timestamp uint64,
	hub *Hub,
	syncer *FrameSyncer,
) {
	metrics.RecordFrameIn(len(data))

	// Record edge-to-backend latency from SASP header timestamp
	metrics.RecordLatency(timestamp)

	var img image.Image
	var err error

	switch fType {
	case TypeBackground:
		img, err = jpeg.Decode(bytes.NewReader(data))
	case TypeROI:
		// Auto-detect format: supports PNG and WebP via registered decoders
		img, _, err = image.Decode(bytes.NewReader(data))
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
			// BG-only frame (roiCount == 0)
			encodeAndBuffer(result, false)
		}
	case TypeROI:
		syncer.AddROI(frameID, img, x, y, roiCount)
		if roiCount > 0 {
			metrics.RecordPersons(roiCount)
		}
	}
}

func encodeAndBuffer(img image.Image, hasROIs bool) {
	var out bytes.Buffer
	if err := jpeg.Encode(&out, img, &jpeg.Options{Quality: 82}); err != nil {
		log.Printf("[encode] %v", err)
		return
	}
	smoothBuffer.Push(out.Bytes(), hasROIs)
}

// encodeAndBroadcast is called by FrameSyncer when a composited frame is ready.
// These frames have ROIs composited on top of BG.
func encodeAndBroadcast(img image.Image, hub *Hub) {
	encodeAndBuffer(img, true) // true = has ROI compositing
}

// ─────────────────────────────────────────────
//  CORS middleware
// ─────────────────────────────────────────────

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func jsonResponse(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	envelope := map[string]interface{}{
		"status":    "ok",
		"data":      data,
		"timestamp": time.Now().UnixMilli(),
	}
	json.NewEncoder(w).Encode(envelope)
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

func main() {
	hub := NewHub()
	reassembler := NewReassembler()
	syncer := NewFrameSyncer(hub)
	smoothBuffer = NewSmoothBuffer(hub)

	go hub.run()

	// ── API Routes ───────────────────────────────────────────────────────────

	// Serve static frontend
	http.Handle("/", http.FileServer(http.Dir("./public")))

	// Video stream — WebSocket (backward-compatible + new API path)
	http.HandleFunc("/ws", hub.serveWS)
	http.HandleFunc("/api/stream", hub.serveWS)

	// Metrics — robust server-side computed metrics
	http.HandleFunc("/api/metrics", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		snap := metrics.GetSnapshot()
		jsonResponse(w, snap)
	}))

	// Legacy /stats endpoint — redirects to new API for backward compat
	http.HandleFunc("/stats", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		snap := metrics.GetSnapshot()
		jsonResponse(w, snap)
	}))

	// Health check
	http.HandleFunc("/api/health", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, map[string]interface{}{
			"healthy":        true,
			"uptime_seconds": time.Since(metrics.startTime).Seconds(),
			"version":        "4.0",
		})
	}))

	// Config — current system configuration
	http.HandleFunc("/api/config", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, map[string]interface{}{
			"roi_fallback_window_ms": ROIFallbackWindow.Milliseconds(),
			"broadcast_buffer_size":  BroadcastBufferSize,
			"output_jpeg_quality":    82,
			"udp_listen":             "127.0.0.1:5000",
			"http_listen":            ":8080",
		})
	}))

	go func() {
		log.Println("[http] → http://localhost:8080")
		log.Println("[api]  → /api/stream, /api/metrics, /api/health, /api/config")
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

	fmt.Println("════════════════════════════════════════════════════")
	fmt.Println("  SASP Backend v4  —  http://localhost:8080")
	fmt.Println("  API             —  /api/stream  /api/metrics")
	fmt.Println("                     /api/health  /api/config")
	fmt.Println("════════════════════════════════════════════════════")

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
		fType := buf[5]
		fID := binary.BigEndian.Uint32(buf[6:10])
		seq := binary.BigEndian.Uint16(buf[10:12])
		total := binary.BigEndian.Uint16(buf[12:14])
		timestamp := binary.BigEndian.Uint64(buf[14:22])
		roiCount := int(buf[22])
		xPos := int(binary.BigEndian.Uint16(buf[24:26]))
		yPos := int(binary.BigEndian.Uint16(buf[26:28]))

		// Use pooled slice instead of heap allocation per packet
		payloadSize := n - HeaderSize
		payload := getPooledSlice(payloadSize)
		copy(payload, buf[HeaderSize:n])

		full, x, y, rc, ts, ok := reassembler.AddPart(
			fID, uint32(roiCount), uint32(buf[23]),
			fType, int(seq), int(total),
			payload, xPos, yPos, timestamp,
		)
		if ok {
			go processFrame(full, fType, fID, rc, x, y, ts, hub, syncer)
		}
	}
}

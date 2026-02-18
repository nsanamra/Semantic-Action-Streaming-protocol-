package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ─────────────────────────────────────────────
//  Protocol Constants
// ─────────────────────────────────────────────

const (
	HeaderSize          = 28
	FrameTypeBackground = 0
	FrameTypeROI        = 1

	// How long to wait for a paired ROI before rendering background alone
	ROIWaitTimeout = 80 * time.Millisecond

	// Broadcast channel buffer — frames beyond this are dropped for slow clients
	BroadcastBufferSize = 8
)

// ─────────────────────────────────────────────
//  Frame Sync Buffer
//
//  Holds a background canvas for frameID N and
//  accumulates every ROI that Python sends for
//  that same frame (one per detected person).
//
//  Python tells us up-front (via fClass) whether
//  ANY objects were detected, but NOT how many.
//  So instead of waiting for a fixed count we use
//  a short collection window: after the first ROI
//  lands we wait up to ROICollectWindow for more,
//  then flush whatever we have.
//
//  Timeline for a 2-person frame:
//    t=0ms   BG arrives  → stored, timer not started yet
//    t=2ms   ROI-A lands → stitched onto canvas, timer starts
//    t=4ms   ROI-B lands → stitched onto canvas
//    t=6ms   (collect window still open — no flush yet)
//    t=12ms  collect window expires → flush final canvas to hub
// ─────────────────────────────────────────────

const ROICollectWindow = 12 * time.Millisecond // wait this long after first ROI for siblings

type pendingFrame struct {
	canvas    *image.RGBA // grows as each ROI is stitched in
	bg        image.Image // original background (needed for first stitch)
	arrivedAt time.Time   // when BG arrived
	firstROI  time.Time   // when first ROI arrived (zero = none yet)
	roiCount  int         // how many ROIs have been stitched so far
	hasROI    bool        // did Python signal objects exist?
	flushed   bool        // guard: only flush once
}

type FrameSyncer struct {
	mu      sync.Mutex
	pending map[uint32]*pendingFrame
	hub     *Hub // needed so gcLoop can flush timed-out frames
}

func NewFrameSyncer(hub *Hub) *FrameSyncer {
	fs := &FrameSyncer{pending: make(map[uint32]*pendingFrame), hub: hub}
	go fs.gcLoop()
	return fs
}

// StoreBG records an arrived background. Returns (bg, true) immediately
// when Python signals no objects are in the frame (fast path).
func (fs *FrameSyncer) StoreBG(frameID uint32, bg image.Image, expectROI bool) (image.Image, bool) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if !expectROI {
		delete(fs.pending, frameID)
		return bg, true // emit background directly — nobody to stitch
	}

	// Build initial RGBA canvas from the background so every incoming
	// ROI can be draw.Over'd onto it incrementally.
	bounds := bg.Bounds()
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, bg, image.Point{}, draw.Src)

	fs.pending[frameID] = &pendingFrame{
		canvas:    canvas,
		bg:        bg,
		arrivedAt: time.Now(),
		hasROI:    true,
	}
	return nil, false
}

// AddROI stitches one person's ROI tile onto the shared canvas for frameID.
// After the first ROI lands it arms a timer; when the timer fires the
// canvas (with all persons painted) is broadcast.
func (fs *FrameSyncer) AddROI(frameID uint32, roi image.Image, x, y int) {
	fs.mu.Lock()

	pf, ok := fs.pending[frameID]
	if !ok || pf.flushed {
		// Background not yet arrived or frame already emitted — discard.
		fs.mu.Unlock()
		return
	}

	// Clip destination rect to canvas bounds (safety for edge-case coords)
	bounds := pf.canvas.Bounds()
	dstRect := image.Rect(x, y, x+roi.Bounds().Dx(), y+roi.Bounds().Dy()).Intersect(bounds)
	if !dstRect.Empty() {
		draw.Draw(pf.canvas, dstRect, roi, image.Point{}, draw.Over)
	}
	pf.roiCount++

	isFirst := pf.firstROI.IsZero()
	if isFirst {
		pf.firstROI = time.Now()
	}
	fs.mu.Unlock()

	// Arm the collection-window timer only on the FIRST ROI for this frame.
	// Subsequent ROIs for the same frame just paint onto the canvas; the
	// timer goroutine will flush the finished result.
	if isFirst {
		go fs.flushAfterWindow(frameID)
	}
}

// flushAfterWindow waits for the collect window, then broadcasts the canvas.
func (fs *FrameSyncer) flushAfterWindow(frameID uint32) {
	time.Sleep(ROICollectWindow)
	fs.flushFrame(frameID)
}

func (fs *FrameSyncer) flushFrame(frameID uint32) {
	fs.mu.Lock()
	pf, ok := fs.pending[frameID]
	if !ok || pf.flushed {
		fs.mu.Unlock()
		return
	}
	pf.flushed = true
	canvas := pf.canvas
	delete(fs.pending, frameID)
	fs.mu.Unlock()

	log.Printf("[sync] flushing frame %d (%d ROIs stitched)", frameID, pf.roiCount)
	encodeAndBroadcast(canvas, fs.hub)
}

// gcLoop evicts frames that never received any ROI within the timeout window.
func (fs *FrameSyncer) gcLoop() {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		fs.mu.Lock()
		now := time.Now()
		var toFlush []uint32
		for id, pf := range fs.pending {
			if !pf.flushed && now.Sub(pf.arrivedAt) > ROIWaitTimeout {
				toFlush = append(toFlush, id)
			}
		}
		fs.mu.Unlock()

		for _, id := range toFlush {
			log.Printf("[sync] frame %d timed out waiting for ROI — flushing BG", id)
			fs.flushFrame(id)
		}
	}
}

// ─────────────────────────────────────────────
//  Reassembler
//  Composite key prevents BG/ROI cross-contamination.
// ─────────────────────────────────────────────

type ChunkKey struct {
	ID   uint32
	Type byte
}

type FrameBuffer struct {
	Parts    [][]byte
	Received int
	Total    int
	X, Y     int
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

func (r *Reassembler) AddPart(id uint32, fType byte, seq, total int, data []byte, x, y int) ([]byte, int, int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := ChunkKey{ID: id, Type: fType}
	fb, ok := r.frames[key]
	if !ok {
		fb = &FrameBuffer{Parts: make([][]byte, total), Total: total}
		r.frames[key] = fb
	}

	if seq < len(fb.Parts) && fb.Parts[seq] == nil {
		fb.Parts[seq] = data
		fb.Received++
		fb.X = x
		fb.Y = y
	}

	if fb.Received == fb.Total {
		full := bytes.Join(fb.Parts, nil)
		delete(r.frames, key)
		return full, fb.X, fb.Y, true
	}
	return nil, 0, 0, false
}

// gcLoop evicts stale incomplete frames (e.g. packets lost in transit)
func (r *Reassembler) gcLoop() {
	// We track insertion time via a side map
	type entry struct{ t time.Time }
	timestamps := make(map[ChunkKey]time.Time)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		r.mu.Lock()
		now := time.Now()
		for k := range r.frames {
			if _, seen := timestamps[k]; !seen {
				timestamps[k] = now
			} else if now.Sub(timestamps[k]) > 2*time.Second {
				delete(r.frames, k)
				delete(timestamps, k)
				log.Printf("[reassembler] evicted stale frame key=%+v", k)
			}
		}
		// Clean up timestamps for keys no longer in frames
		for k := range timestamps {
			if _, ok := r.frames[k]; !ok {
				delete(timestamps, k)
			}
		}
		r.mu.Unlock()
	}
}

// ─────────────────────────────────────────────
//  WebSocket Hub
//  Thread-safe client registry + slow-client drop.
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
			log.Printf("[hub] client connected (total=%d)", len(h.clients))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				c.closeOnce.Do(func() { close(c.send) })
			}
			h.mu.Unlock()
			log.Printf("[hub] client disconnected (total=%d)", len(h.clients))

		case frame := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- frame:
				default:
					// Slow client — drop frame rather than blocking the pipeline
					log.Printf("[hub] dropping frame for slow client")
				}
			}
			h.mu.RUnlock()
		}
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin:     func(r *http.Request) bool { return true },
	ReadBufferSize:  1024,
	WriteBufferSize: 32 * 1024,
}

func (h *Hub) serveWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[ws] upgrade error: %v", err)
		return
	}
	c := &client{conn: conn, send: make(chan []byte, 4)}
	h.register <- c
	go c.writePump()

	// Read loop — detects client disconnect
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
//  Frame Processing
// ─────────────────────────────────────────────

func processFrame(data []byte, fType byte, frameID uint32, fClass byte, x, y int, hub *Hub, syncer *FrameSyncer) {
	var img image.Image
	var err error

	if fType == FrameTypeBackground {
		img, err = jpeg.Decode(bytes.NewReader(data))
	} else if fType == FrameTypeROI {
		img, err = png.Decode(bytes.NewReader(data))
	}

	if err != nil {
		log.Printf("[process] decode error fType=%d frameID=%d: %v", fType, frameID, err)
		return
	}

	if fType == FrameTypeBackground {
		expectROI := fClass != 0
		result, ready := syncer.StoreBG(frameID, img, expectROI)
		if ready {
			// No ROI expected — emit background directly
			encodeAndBroadcast(result, hub)
		}

	} else if fType == FrameTypeROI {
		// Paint this person's tile onto the shared canvas for this frame.
		// The syncer will broadcast automatically once the collect window closes.
		syncer.AddROI(frameID, img, x, y)
	}
}

func encodeAndBroadcast(img image.Image, hub *Hub) {
	// Encode OUTSIDE any lock — this is the slow step (~5-15ms)
	var out bytes.Buffer
	if err := jpeg.Encode(&out, img, &jpeg.Options{Quality: 85}); err != nil {
		log.Printf("[encode] jpeg error: %v", err)
		return
	}

	select {
	case hub.broadcast <- out.Bytes():
	default:
		log.Printf("[broadcast] channel full, dropping frame")
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

	// Static file server
	http.Handle("/", http.FileServer(http.Dir("./public")))
	http.HandleFunc("/ws", hub.serveWS)

	go func() {
		log.Println("[http] Listening on :8080")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("[http] fatal: %v", err)
		}
	}()

	// UDP listener
	addr, err := net.ResolveUDPAddr("udp", "127.0.0.1:5000")
	if err != nil {
		log.Fatalf("[udp] resolve error: %v", err)
	}
	udpConn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("[udp] listen error: %v", err)
	}
	defer udpConn.Close()

	fmt.Println("════════════════════════════════════════")
	fmt.Println("  SASP Backend  —  http://localhost:8080 ")
	fmt.Println("════════════════════════════════════════")

	buf := make([]byte, 65535)
	for {
		n, _, err := udpConn.ReadFromUDP(buf)
		if err != nil {
			log.Printf("[udp] read error: %v", err)
			continue
		}
		if n < HeaderSize {
			continue
		}

		// Parse 28-byte header
		// Offset: 0-3 Magic, 4 Version, 5 Type, 6-9 FrameID, 10-11 Seq,
		//         12-13 Total, 14-21 Timestamp, 22 Class, 23 Priority, 24-25 X, 26-27 Y
		fType := buf[5]
		fID := binary.BigEndian.Uint32(buf[6:10])
		seq := binary.BigEndian.Uint16(buf[10:12])
		total := binary.BigEndian.Uint16(buf[12:14])
		fClass := buf[22]
		xPos := int(binary.BigEndian.Uint16(buf[24:26]))
		yPos := int(binary.BigEndian.Uint16(buf[26:28]))

		payload := make([]byte, n-HeaderSize)
		copy(payload, buf[HeaderSize:n])

		if fullFrame, x, y, ok := reassembler.AddPart(fID, fType, int(seq), int(total), payload, xPos, yPos); ok {
			go processFrame(fullFrame, fType, fID, fClass, x, y, hub, syncer)
		}
	}
}

// Package reconstruction handles frame reassembly and stitching for the SASP pipeline.
package reconstruction

import (
	"bytes"
	"sync"
	"time"
)

// ─────────────────────────────────────────────
//  Reassembler
//  Collects UDP chunks for a given (frameID, frameType)
//  pair and emits the complete payload once all parts arrive.
//
//  Key design decisions:
//   • Composite key (ID + Type) prevents BG and ROI packets
//     with the same frame ID from overwriting each other.
//   • A GC loop evicts incomplete frames whose packets were
//     lost in transit, preventing unbounded memory growth.
// ─────────────────────────────────────────────

// ChunkKey uniquely identifies one logical stream chunk.
type ChunkKey struct {
	ID   uint32
	Type byte
}

// frameBuffer accumulates chunks for one logical frame.
type frameBuffer struct {
	parts     [][]byte
	received  int
	total     int
	x, y      int
	createdAt time.Time
}

// Reassembler is safe for concurrent use from multiple goroutines.
type Reassembler struct {
	mu     sync.Mutex
	frames map[ChunkKey]*frameBuffer
}

// New returns an initialised Reassembler with an active GC goroutine.
func New() *Reassembler {
	r := &Reassembler{frames: make(map[ChunkKey]*frameBuffer)}
	go r.gcLoop()
	return r
}

// AddPart records one UDP chunk. When the last expected chunk arrives the
// full reassembled payload is returned along with its canvas coordinates.
func (r *Reassembler) AddPart(
	id uint32,
	fType byte,
	seq, total int,
	data []byte,
	x, y int,
) (payload []byte, ox, oy int, complete bool) {

	r.mu.Lock()
	defer r.mu.Unlock()

	key := ChunkKey{ID: id, Type: fType}

	fb, ok := r.frames[key]
	if !ok {
		fb = &frameBuffer{
			parts:     make([][]byte, total),
			total:     total,
			createdAt: time.Now(),
		}
		r.frames[key] = fb
	}

	// Guard against out-of-range seq numbers from malformed packets
	if seq < 0 || seq >= len(fb.parts) {
		return nil, 0, 0, false
	}

	// Deduplicate retransmissions
	if fb.parts[seq] == nil {
		fb.parts[seq] = data
		fb.received++
		fb.x = x
		fb.y = y
	}

	if fb.received == fb.total {
		full := bytes.Join(fb.parts, nil)
		delete(r.frames, key)
		return full, fb.x, fb.y, true
	}

	return nil, 0, 0, false
}

// gcLoop periodically evicts incomplete frames whose packets were lost.
// Stale frame threshold is 2 s — well beyond any realistic 30 fps window.
const staleThreshold = 2 * time.Second

func (r *Reassembler) gcLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		r.mu.Lock()
		now := time.Now()
		for k, fb := range r.frames {
			if now.Sub(fb.createdAt) > staleThreshold {
				delete(r.frames, k)
			}
		}
		r.mu.Unlock()
	}
}

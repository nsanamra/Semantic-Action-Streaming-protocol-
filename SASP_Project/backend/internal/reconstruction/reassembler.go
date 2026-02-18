package reconstruction

import (
	"sync"
	"time"
)

type FrameBuffer struct {
	Parts      [][]byte
	Received   []bool
	Total      int
	LastUpdate time.Time
}

type Reassembler struct {
	mu     sync.Mutex
	frames map[uint32]*FrameBuffer
}

func (r *Reassembler) AddPart(frameID uint32, seq uint16, total uint16, data []byte) ([]byte, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	fb, exists := r.frames[frameID]
	if !exists {
		fb = &FrameBuffer{
			Parts:    make([][]byte, total),
			Received: make([]bool, total),
			Total:    int(total),
		}
		r.frames[frameID] = fb
	}

	fb.Parts[seq] = data
	fb.Received[seq] = true
	fb.LastUpdate = time.Now()

	// Check if all parts for this frame arrived
	for _, rec := range fb.Received {
		if !rec {
			return nil, false
		}
	}

	// Reassemble full byte stream
	var fullFrame []byte
	for _, p := range fb.Parts {
		fullFrame = append(fullFrame, p...)
	}
	delete(r.frames, frameID) // Clear memory
	return fullFrame, true
}

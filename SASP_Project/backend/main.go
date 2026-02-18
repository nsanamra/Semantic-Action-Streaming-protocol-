package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"net"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

const (
	HeaderSize = 24
	UDPPort    = 5000
	HTTPPort   = ":8080"
)

// Global state for the video stream
var (
	currentBackground image.Image
	bgMutex           sync.Mutex
	upgrader          = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
)

// Reassembler manages incoming UDP packet chunks
type Reassembler struct {
	mu     sync.Mutex
	frames map[uint32]*FrameBuffer
}

type FrameBuffer struct {
	Parts    [][]byte
	Received int
	Total    int
}

func (r *Reassembler) AddPart(id uint32, seq, total int, data []byte) ([]byte, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.frames[id]; !ok {
		r.frames[id] = &FrameBuffer{Parts: make([][]byte, total), Total: total}
	}

	fb := r.frames[id]
	if fb.Parts[seq] == nil {
		fb.Parts[seq] = data
		fb.Received++
	}

	if fb.Received == fb.Total {
		full := bytes.Join(fb.Parts, nil)
		delete(r.frames, id)
		return full, true
	}
	return nil, false
}

type Hub struct {
	clients   map[*websocket.Conn]bool
	broadcast chan []byte
}

func main() {
	hub := &Hub{clients: make(map[*websocket.Conn]bool), broadcast: make(chan []byte)}
	reassembler := &Reassembler{frames: make(map[uint32]*FrameBuffer)}
	go hub.run()

	// 1. WebSocket & Static File Server
	http.Handle("/", http.FileServer(http.Dir("./public")))
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, _ := upgrader.Upgrade(w, r, nil)
		hub.clients[conn] = true
	})
	go http.ListenAndServe(HTTPPort, nil)

	// 2. UDP Receiver
	addr, _ := net.ResolveUDPAddr("udp", "127.0.0.1:5000")
	conn, _ := net.ListenUDP("udp", addr)
	fmt.Println("SASP Backend Live at http://localhost:8080")

	buf := make([]byte, 2048)
	for {
		n, _, _ := conn.ReadFromUDP(buf)
		if n < HeaderSize {
			continue
		}

		// Parse Header
		fType := buf[5]
		fID := binary.BigEndian.Uint32(buf[6:10])
		seq := binary.BigEndian.Uint16(buf[10:12])
		total := binary.BigEndian.Uint16(buf[12:14])
		payload := make([]byte, n-HeaderSize)
		copy(payload, buf[HeaderSize:n])

		// Reassemble and Process
		if fullFrame, ok := reassembler.AddPart(fID, int(seq), int(total), payload); ok {
			processFrame(fullFrame, fType, hub)
		}
	}
}

func processFrame(data []byte, fType byte, hub *Hub) {
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return
	}

	bgMutex.Lock()
	defer bgMutex.Unlock()

	if fType == 0 { // Background
		currentBackground = img
	} else if fType == 1 && currentBackground != nil { // ROI
		// For this step, we'll overlay the ROI in the center
		// In a later step, we'll extract (x, y) from the SASP header
		b := currentBackground.Bounds()
		final := image.NewRGBA(b)
		draw.Draw(final, b, currentBackground, image.Point{}, draw.Src)

		roiOffset := image.Pt(b.Dx()/2-img.Bounds().Dx()/2, b.Dy()/2-img.Bounds().Dy()/2)
		draw.Draw(final, img.Bounds().Add(roiOffset), img, image.Point{}, draw.Over)
		currentBackground = final
	}

	// Send to Dashboard
	var out bytes.Buffer
	jpeg.Encode(&out, currentBackground, nil)
	hub.broadcast <- out.Bytes()
}

func (h *Hub) run() {
	for frame := range h.broadcast {
		for client := range h.clients {
			client.WriteMessage(websocket.BinaryMessage, frame)
		}
	}
}

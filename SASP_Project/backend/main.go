package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"net"
	"net/http"
	"sync"

	"sasp-backend/internal/reconstruction" // Update to your module name

	"github.com/gorilla/websocket"
)

const HeaderSize = 28

var (
	currentBackground image.Image
	bgMutex           sync.Mutex
	upgrader          = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
)

// FIX 1: Create a composite key so Background and ROI packets don't overwrite each other
type ChunkKey struct {
	ID   uint32
	Type byte
}

type Reassembler struct {
	mu     sync.Mutex
	frames map[ChunkKey]*FrameBuffer
}

type FrameBuffer struct {
	Parts    [][]byte
	Received int
	Total    int
	X, Y     int
}

func (r *Reassembler) AddPart(id uint32, fType byte, seq, total int, data []byte, x, y int) ([]byte, int, int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := ChunkKey{ID: id, Type: fType}

	if _, ok := r.frames[key]; !ok {
		r.frames[key] = &FrameBuffer{Parts: make([][]byte, total), Total: total}
	}

	fb := r.frames[key]
	if fb.Parts[seq] == nil {
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

type Hub struct {
	clients   map[*websocket.Conn]bool
	broadcast chan []byte
}

func main() {
	hub := &Hub{clients: make(map[*websocket.Conn]bool), broadcast: make(chan []byte)}
	reassembler := &Reassembler{frames: make(map[ChunkKey]*FrameBuffer)}
	go hub.run()

	http.Handle("/", http.FileServer(http.Dir("./public")))
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, _ := upgrader.Upgrade(w, r, nil)
		hub.clients[conn] = true
	})
	go http.ListenAndServe(":8080", nil)

	addr, _ := net.ResolveUDPAddr("udp", "127.0.0.1:5000")
	conn, _ := net.ListenUDP("udp", addr)
	fmt.Println("SASP Backend Live (Sync Fixed) at http://localhost:8080")

	buf := make([]byte, 2048)
	for {
		n, _, _ := conn.ReadFromUDP(buf)
		if n < HeaderSize {
			continue
		}

		fType := buf[5]
		fID := binary.BigEndian.Uint32(buf[6:10])
		seq := binary.BigEndian.Uint16(buf[10:12])
		total := binary.BigEndian.Uint16(buf[12:14])

		// Extract roi_count from the Class byte
		fClass := buf[22]
		xPos := int(binary.BigEndian.Uint16(buf[24:26]))
		yPos := int(binary.BigEndian.Uint16(buf[26:28]))

		payload := make([]byte, n-HeaderSize)
		copy(payload, buf[HeaderSize:n])

		// Pass fType into AddPart
		if fullFrame, x, y, ok := reassembler.AddPart(fID, fType, int(seq), int(total), payload, xPos, yPos); ok {
			processFrame(fullFrame, fType, fClass, x, y, hub)
		}
	}
}

func processFrame(data []byte, fType byte, fClass byte, x, y int, hub *Hub) {
	var img image.Image
	var err error

	if fType == 0 {
		img, err = jpeg.Decode(bytes.NewReader(data))
	} else if fType == 1 {
		img, err = png.Decode(bytes.NewReader(data))
	}

	if err != nil {
		return
	}

	bgMutex.Lock()
	defer bgMutex.Unlock()

	if fType == 0 {
		currentBackground = img
		// FIX 2: Only broadcast the raw background if Python reported 0 objects detected
		if fClass == 0 {
			var out bytes.Buffer
			jpeg.Encode(&out, currentBackground, nil)
			hub.broadcast <- out.Bytes()
		}
	} else if fType == 1 && currentBackground != nil {
		stitched := reconstruction.Stitch(currentBackground, img, x, y)
		var out bytes.Buffer
		jpeg.Encode(&out, stitched, nil)
		hub.broadcast <- out.Bytes()
	}
}

func (h *Hub) run() {
	for frame := range h.broadcast {
		for client := range h.clients {
			client.WriteMessage(websocket.BinaryMessage, frame)
		}
	}
}

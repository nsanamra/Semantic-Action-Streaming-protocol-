# SASP — Semantic Adaptive Streaming Protocol

> A privacy-first, bandwidth-efficient real-time video pipeline that uses YOLO segmentation to transmit only the pixels that matter — **~3–5× less bandwidth than traditional JPEG streaming**.

---

## What Is SASP?

Traditional video streaming sends every pixel of every frame at full quality. SASP treats the stream semantically:

- The **background** is blurred and sent at low quality (~2–3 KB/frame)
- **Each detected person** is segmented, sharpened, and sent as a high-quality RGBA tile (~4–8 KB/tile)
- The Go backend **composites** both streams and pushes a final JPEG to browsers over WebSocket

The result: sharp subjects, blurred backgrounds, dramatically lower bandwidth, and built-in privacy.

```
Camera → YOLO seg → BG (blurred, low-Q JPEG) ─┐
                  → ROI tiles (sharp PNG/BGRA) ─┴→ Go → Composite → Browser
```

---

## Repository Structure

```
SASP_Project/
├── edge/                        # Python transmitter (runs on camera device)
│   ├── main.py                  # Entry point — pipelined camera grab + worker
│   ├── requirements.txt         # Python dependencies
│   └── scripts/
│       ├── detector.py          # YOLOv8-seg wrapper (detection + BG blur)
│       └── transmitter.py       # Standalone demo mode (no YOLO needed)
│
├── backend/                     # Go backend (receives UDP, composites, serves browser)
│   ├── main.go                  # UDP server, Reassembler, FrameSyncer, WebSocket hub
│   ├── go.mod / go.sum
│   ├── public/
│   │   └── index.html           # SASP browser monitor (port 8080)
│   └── internal/reconstruction/
│       ├── reassembler.go       # Chunk → full payload reassembly
│       └── stitcher.go          # draw.Over compositor
│
└── traditional/                 # Baseline comparison system
    ├── traditional_streamer.py  # Naive full-frame JPEG streamer (port 5001)
    ├── traditional_backend.go   # Go backend for traditional stream (port 8081)
    ├── go.mod / go.sum
    └── public_trad/
        └── index.html           # Traditional browser monitor
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | |
| Go | ≥ 1.21 | |
| Webcam | — | USB or built-in, index 0 by default |
| CUDA (optional) | ≥ 11.8 | Recommended for real-time YOLO inference |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SASP_Project.git
cd SASP_Project
```

### 2. Set up the Python edge environment

```bash
cd edge

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python-headless numpy

# Download the YOLO segmentation model
# Option A — auto-download on first run (requires internet)
# Option B — manual download:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt
# Then set the env variable so detector.py finds it:
export YOLO_MODEL=yolov8s-seg.pt
```

> **No GPU?** The demo transmitter (`transmitter.py`) runs without YOLO at all — it uses a centre-crop mock ROI. See [Demo Mode](#demo-mode-no-yolo-required).

### 3. Set up the Go backend

```bash
cd backend
go mod download
go build -o sasp-backend .
```

### 4. Set up the traditional baseline (optional)

```bash
cd traditional
go mod download
go build -o traditional-backend traditional_backend.go
```

---

## Running SASP

Run each component in a **separate terminal**, in order.

### Terminal 1 — Go Backend

```bash
cd backend
./sasp-backend
# ════════════════════════════════════════════════════════
#   SASP Backend v3  —  http://localhost:8080
#   Stats           —  http://localhost:8080/stats
# ════════════════════════════════════════════════════════
```

### Terminal 2 — Python Edge Transmitter

**Full mode (with YOLO):**
```bash
cd edge
source venv/bin/activate
export YOLO_MODEL=yolov8s-seg.pt   # skip if using default path
python main.py
```

**Demo mode (no YOLO required):**
```bash
cd edge
source venv/bin/activate
python scripts/transmitter.py
```

### View the stream

Open your browser at **http://localhost:8080**

You should see:
- A live feed with blurred background and sharp person tiles
- Real-time stats: FPS, bandwidth, person count, frame latency

---

## Running the Traditional Baseline (Comparison)

In two additional terminals:

```bash
# Terminal 3 — Traditional Go backend
cd traditional
./traditional-backend

# Terminal 4 — Traditional Python streamer
cd edge
source venv/bin/activate
python traditional/traditional_streamer.py
```

Open **http://localhost:8081** to see the traditional stream.

Compare bandwidth in the stats panels — SASP typically uses **3–5× less**.

---

## Demo Mode (No YOLO Required)

If you don't have a GPU or don't want to install Ultralytics, use the standalone demo transmitter:

```bash
cd edge
python scripts/transmitter.py
```

This transmits:
- A blurred full frame as the background
- A centre-crop (middle 20% of frame) as the mock ROI

The Go backend handles it identically — it's a valid SASP stream.

---

## Configuration

### Edge (`edge/main.py` and `edge/scripts/transmitter.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_IP` | `127.0.0.1` | IP of the Go backend |
| `SERVER_PORT` | `5000` | UDP port |
| `CAMERA_INDEX` | `0` | OpenCV camera index (or RTSP URL string) |
| `FRAME_WIDTH/HEIGHT` | `640 × 480` | Capture resolution |
| `TARGET_FPS` | `30` | Transmit rate cap |
| `BG_JPEG_QUALITY` | `22` | Background JPEG quality (lower = smaller) |
| `PNG_COMPRESSION` | `1` | ROI PNG compression (0=none, 9=max) |
| `YOLO_MODEL` (env) | `yolo26s-seg.pt` | Path to YOLO weights file |

### Detector (`edge/scripts/detector.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `DETECTION_CONFIDENCE` | `0.40` | YOLO confidence threshold |
| `SMOOTHING_ALPHA` | `0.82` | EMA smoothing for bounding boxes |
| `DEADZONE_PX` | `12.0` | Pixel movement below which bbox is frozen |
| `ROI_PAD` | `28` | Padding around each person bbox (px) |
| `MAX_TRACK_AGE` | `90` | Frames before an unseen track is evicted |
| `BG_DIFF_THRESHOLD` | `6.0` | Scene stillness threshold for BG cache reuse |

### Backend (`backend/main.go`)

| Constant | Default | Description |
|----------|---------|-------------|
| `ROIFallbackWindow` | `8ms` | Collect window when ROI count unknown |
| `BroadcastBufferSize` | `32` | WebSocket hub frame buffer per client |
| UDP listen | `:5000` | Receives SASP packets from Python |
| HTTP | `:8080` | Serves browser frontend and WebSocket |

---

## How It Works — Technical Summary

### The SASP Protocol Header (28 bytes, big-endian)

Every UDP packet starts with this fixed header:

```
Offset  Size  Field        Description
0       4     Magic        "SASP" — protocol identifier
4       1     Version      Always 1
5       1     Type         0 = Background JPEG · 1 = ROI PNG tile
6       4     FrameID      uint32, wraps at 2³²
10      2     SeqNum       Chunk index within this payload
12      2     TotalParts   Total chunks for this payload
14      8     Timestamp    Unix nanoseconds (time.time_ns())
22      1     ROICount     ★ Exact number of ROI tiles in this frame
23      1     ROIIndex     Which tile this is (0-based)
24      2     X            Canvas X coordinate of ROI top-left
26      2     Y            Canvas Y coordinate of ROI top-left
```

**★ ROICount** is the key synchronisation field. It lets the Go `FrameSyncer` flush the composited frame the instant all ROI tiles arrive — no timeout polling needed.

### Pipeline Summary

```
[Camera grab — main thread]
    ↓  (queue, drops oldest if worker is behind)
[Worker thread]
    ↓  detector.detect(frame)
    ├─→ blurred BG → JPEG encode → UDP send (Type=0, ROICount=N)
    └─→ per-person BGRA tile → PNG encode (parallel) → UDP send (Type=1)
    
[Go — Reassembler]
    Collects 1400-byte MTU chunks per (FrameID, Type) pair
    → emits full payload when all chunks arrive
    
[Go — FrameSyncer]
    Strategy A (preferred): wait for exactly ROICount ROI tiles → flush
    Strategy B (fallback):  8ms window after first ROI → flush
    
[Go — Compositor]
    draw.Draw(canvas, bg, draw.Src)         ← paint background
    draw.Draw(canvas, roi, draw.Over)       ← blend per-person alpha
    → JPEG encode at quality 82 → WebSocket broadcast
    
[Browser]
    WebSocket binary → createImageBitmap → requestAnimationFrame → canvas
```

### Bandwidth Comparison

| Metric | Traditional | SASP |
|--------|-------------|------|
| Frame size | ~18–25 KB | ~4–11 KB (1 person) |
| Background quality | 80% JPEG (full) | 22% JPEG + blur |
| Bandwidth | ~540–750 KB/s | ~120–250 KB/s |
| Useful data ratio | ~20–30% | ~85–95% |
| Privacy | Full scene | Background blurred |

---

## Ports Reference

| Port | Protocol | Used by |
|------|----------|---------|
| 5000 | UDP | SASP: Python → Go backend |
| 5001 | UDP | Traditional: Python → Go backend |
| 8080 | HTTP/WS | SASP browser monitor |
| 8081 | HTTP/WS | Traditional browser monitor |

---

## Troubleshooting

**Camera not found**
```bash
# List available cameras
python3 -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]"
# Then set CAMERA_INDEX in main.py / transmitter.py
```

**YOLO model not found**
```bash
# Download manually and set the env variable
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt
export YOLO_MODEL=$(pwd)/yolov8s-seg.pt
```

**Port already in use**
```bash
# Find what's using port 5000
lsof -i :5000
# Or change SERVER_PORT in main.py and the UDP listen address in main.go
```

**Slow inference / dropped frames**
- Switch to a smaller model: `yolov8n-seg.pt` (nano)
- Lower `TARGET_FPS` in `main.py`
- Reduce capture resolution: `FRAME_WIDTH=320`, `FRAME_HEIGHT=240`
- Lower `DETECTION_CONFIDENCE` to 0.35 (fewer false negatives, slightly more CPU)

**Black screen in browser**
- Confirm Go backend is running: `curl http://localhost:8080/stats`
- Confirm Python transmitter is running and no UDP errors in its output
- Check firewall isn't blocking UDP 5000

---

## Dependencies

### Python
- `ultralytics` — YOLOv8 inference + segmentation
- `opencv-python-headless` — camera capture, JPEG/PNG encode, image processing
- `numpy` — mask processing, EMA smoothing

### Go
- `github.com/gorilla/websocket` — WebSocket server
- `github.com/disintegration/imaging` — image utilities
- `golang.org/x/image` — extended image format support

---

## License

MIT — see `LICENSE` for details.
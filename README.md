# SASP — Semantic Adaptive Streaming Protocol

> **Stream smarter, not harder.** SASP detects people in real-time, transmits only what matters at full quality, and blurs everything else — reducing bandwidth by up to 50% with zero perceptible quality loss on subjects.

---

## Table of Contents

- [What is SASP?](#what-is-sasp)
- [How It Works](#how-it-works)
- [Bandwidth Savings](#bandwidth-savings)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Running the Comparison Demo](#running-the-comparison-demo)
- [Viewing the Stream](#viewing-the-stream)
- [Configuration](#configuration)
- [Architecture Deep Dive](#architecture-deep-dive)
- [The Protocol Header](#the-protocol-header)
- [Metrics Explained](#metrics-explained)
- [Troubleshooting](#troubleshooting)

---

## What is SASP?

Traditional video streaming sends every pixel of every frame at the same quality — whether it's a person's face or an empty corridor wall. That's wasteful.

**SASP splits every frame into two paths:**

| Path | Content | Quality | Size |
|------|---------|---------|------|
| Background | Blurred full frame | 10% JPEG | ~4 KB |
| ROI tiles | Each detected person | PNG with alpha | ~10 KB each |

The Go backend **stitches** them back together — sharp people on a blurred background — and pushes the final frame to the browser over WebSocket.

**The result:** same 30 FPS stream, dramatically less data.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         EDGE (Python)                           │
│                                                                 │
│  Camera → YOLO Detect → ┬─ Blurred BG  (10% JPEG) ──────────┐  │
│                         └─ Per-Person ROI (PNG+alpha) ──────┐ │  │
└──────────────────────────────────────────────────────────────┼─┼──┘
                                                               │ │
                              UDP port 5000                    │ │
                                                               ▼ ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BACKEND (Go)                             │
│                                                                 │
│  UDP Listener → Reassembler → FrameSyncer → Stitch → WebSocket │
│                                                    ↓            │
│                                              /stats JSON        │
└─────────────────────────────────────────────────────────────────┘
                                                    │
                              WebSocket port 8080   │
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BROWSER (HTML)                            │
│                                                                 │
│  WebSocket → ImageBitmap decode → requestAnimationFrame → Draw  │
│  fetch('/stats') every 1s → Live metrics dashboard             │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-step frame lifecycle

1. **Python grabs a frame** from the webcam
2. **YOLO (YOLOv11-seg)** runs inference — persons only (`class=0`)
3. **Background path**: frame is blurred with Gaussian blur, JPEG encoded at 10% quality (~4 KB), sent over UDP as Type `0` packet
4. **ROI path**: for each detected person, a padded bounding box crop is extracted, sharpened, given a feathered alpha mask from the segmentation mask, and PNG-encoded (~10 KB). Sent as Type `1` packet with exact canvas X/Y coordinates
5. **Go reassembles** multi-chunk UDP packets back into complete frames
6. **FrameSyncer** waits for all ROI tiles for a given frame ID (exact count embedded in header), then stitches them onto the background canvas
7. **Final JPEG** is broadcast over WebSocket to all connected browsers
8. **Browser** decodes with `createImageBitmap` (GPU thread) and draws via `requestAnimationFrame`

---

## Bandwidth Savings

Measured at 640×480, 30 FPS:

| Scenario | Traditional | SASP | Saving |
|----------|------------|------|--------|
| 0 people | ~35 KB/s   | ~4 KB/s | **89%** |
| 1 person | ~35 KB/s   | ~15 KB/s | **57%** |
| 2 people | ~35 KB/s   | ~25 KB/s | **29%** |
| 3 people | ~35 KB/s   | ~35 KB/s | ~0% |

> Beyond ~3 people filling the frame, SASP's ROI tiles approach the size of a full frame. The break-even point depends on how much of the frame the subjects occupy.

---

## Project Structure

```
SASP_Project/
│
├── backend/                        ← Go server
│   ├── internal/
│   │   ├── protocol/               (reserved for future protocol helpers)
│   │   └── reconstruction/
│   │       ├── reassembler.go      UDP chunk → full frame reassembly
│   │       └── stitcher.go         BG + ROI tile compositor
│   ├── public/
│   │   ├── index.html              SASP live monitor (port 8080)
│   │   └── comparison.html         Animated bandwidth comparison demo
│   ├── go.mod
│   ├── go.sum
│   └── main.go                     UDP listener, FrameSyncer, WebSocket hub, /stats
│
├── edge/                           ← Python transmitter
│   ├── models/                     Place your .pt YOLO model here
│   ├── scripts/
│   │   ├── detector.py             YOLOv11-seg inference + mask processing
│   │   └── transmitter.py          Standalone demo (no YOLO needed)
│   └── main.py                     Main edge pipeline (pipelined, per-person)
│
└── traditional/                    ← Baseline comparison system
    ├── public_trad/
    │   └── index.html              Traditional stream monitor (port 8081)
    ├── traditional_backend.go      Go server for traditional stream
    └── traditional_streamer.py     Naive full-frame JPEG streamer
```

---

## Prerequisites

### Python (edge)
- Python 3.10+
- `pip install ultralytics opencv-python`
- A YOLOv11 segmentation model — download from [Ultralytics](https://docs.ultralytics.com):
  ```bash
  # Place in edge/models/
  wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
  mv yolo11s-seg.pt SASP_Project/edge/models/
  ```

### Go (backend)
- Go 1.21+
- Dependencies are in `go.mod`. Run `go mod tidy` once to fetch them:
  ```bash
  cd SASP_Project/backend
  go mod tidy
  ```

### Browser
- Chrome or Firefox (both support `createImageBitmap` and WebSocket binary)

---

## Installation

```bash
# 1. Clone / download the project
cd SASP_Project

# 2. Install Go dependencies
cd backend
go mod tidy
cd ..

# 3. Install Python dependencies
cd edge
pip install ultralytics opencv-python
cd ..

# 4. Download YOLO model (if you don't have one)
wget -P edge/models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
```

---

## Running the System

You need **2 terminals** for the SASP system:

### Terminal 1 — Go Backend
```bash
cd SASP_Project/backend
go run main.go
```
Expected output:
```
════════════════════════════════════════════
  SASP Backend v3  —  http://localhost:8080
  Stats           —  http://localhost:8080/stats
════════════════════════════════════════════
```

### Terminal 2 — Python Edge
```bash
cd SASP_Project/edge
python main.py
```
Expected output:
```
══════════════════════════════════════════════════════
  SASP Edge Transmitter  v3  —  Pipelined + Per-Person
  → 127.0.0.1:5000   target 30 FPS
══════════════════════════════════════════════════════
[sasp] fps=29.8  infer=18.2ms  encode=6.1ms  bw=24.3KB/s  persons=1.0
```

Then open **http://localhost:8080** in your browser.

---

## Running the Comparison Demo

To compare SASP vs Traditional side-by-side with **real camera data**, run 2 more terminals:

### Terminal 3 — Traditional Go Backend
```bash
cd SASP_Project/traditional
go run traditional_backend.go
```

### Terminal 4 — Traditional Python Streamer
```bash
cd SASP_Project/traditional
python traditional_streamer.py
```

Now open **two browser windows side by side:**

| Window | URL | Shows |
|--------|-----|-------|
| Left   | `http://localhost:8080` | SASP stream + metrics |
| Right  | `http://localhost:8081` | Traditional stream + metrics |

Watch the **Bandwidth** metric — same camera, same FPS, dramatically different numbers.

> For an animated simulation without running any code, open `http://localhost:8080/comparison.html` — it shows the bandwidth difference with an interactive person-count slider.

---

## Viewing the Stream

### Main SASP Monitor — `http://localhost:8080`

The dashboard shows:

| Metric | Source | What it means |
|--------|--------|---------------|
| **Receive FPS** | Browser JS | Frames actually rendered per second |
| **Bandwidth** | Browser JS | KB/s received over WebSocket |
| **Frame latency** | Browser JS | Time from WS message → decoded bitmap |
| **Persons** | Go `/stats` | Average people detected per frame |
| **Frames out** | Go `/stats` | Total frames the Go server has broadcast |
| **Dropped** | Go `/stats` | Frames dropped due to slow clients |
| **Clients** | Go `/stats` | Active WebSocket connections |
| **Go FPS** | Go `/stats` | FPS as measured by the Go server |

### Stats API — `http://localhost:8080/stats`

The Go server exposes a JSON endpoint polled by the frontend every second:

```json
{
  "frames_in":      1842,
  "frames_out":     1840,
  "bytes_out":      47382016,
  "dropped_frames": 2,
  "active_clients": 1,
  "avg_fps":        "29.8",
  "avg_persons":    "1.0"
}
```

You can curl it directly:
```bash
curl http://localhost:8080/stats | python -m json.tool
```

---

## Configuration

### Edge (`edge/main.py` — top of file)

```python
SERVER_IP    = "127.0.0.1"   # Go backend IP
SERVER_PORT  = 5000           # Go backend UDP port
CAMERA_INDEX = 0              # Webcam index, or RTSP URL string
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30
BG_JPEG_QUALITY = 10          # Background compression (lower = less bandwidth)
PNG_COMPRESSION = 1           # ROI compression (1=fastest, 9=smallest)
```

### Detector (`edge/scripts/detector.py` — top of file)

```python
DETECTION_CONFIDENCE = 0.40   # YOLO confidence (lower = more detections, more false positives)
SMOOTHING_ALPHA      = 0.82   # BBox smoothing (higher = smoother but more lag)
DEADZONE_PX          = 12.0   # Min movement (px) before bbox updates
ROI_PAD              = 28     # Padding around each person's bounding box
```

### Backend (`backend/main.go` — constants block)

```go
ROIFallbackWindow    = 15 * time.Millisecond  // How long to wait for ROI tiles if count unknown
BroadcastBufferSize  = 12                      // Frame buffer per client before dropping
```

### YOLO model path

Either set the environment variable before running:
```bash
export YOLO_MODEL=models/yolo11s-seg.pt
python main.py
```

Or edit the default in `detector.py`:
```python
MODEL_PATH = os.environ.get("YOLO_MODEL", "models/yolo11s-seg.pt")
```

---

## Architecture Deep Dive

### Python Pipeline (Pipelined — 2 threads)

```
Main thread:  cap.read() → queue.put()   (never stalls)
Worker thread: queue.get() → YOLO → encode → UDP send
```

This separation ensures the camera always captures the latest frame even when YOLO inference takes 20ms. Without it, inference time directly reduces your FPS.

### ROI Count in Protocol Header

The most important architectural decision: **Python tells Go exactly how many ROI tiles are coming** for each frame by embedding the count in the background packet header (`byte 22 = ROICount`).

Go uses this for two strategies:

- **Exact-count flush (fast path):** If `ROICount = 2`, Go waits until exactly 2 ROI tiles arrive, then flushes the canvas immediately. Zero latency tax.
- **Fallback window:** If `ROICount = 0` but `fClass` indicates objects exist (e.g. older client), Go waits 15ms after the first ROI tile for any siblings. This adds a small latency cost but guarantees correctness.

### FrameSyncer

The core of the Go backend. For each incoming frame:

1. Background arrives → RGBA canvas created, pre-filled with blurred BG
2. ROI tile arrives → `draw.Over` composited onto canvas at exact (X, Y)
3. When tile count matches expected → canvas encoded as JPEG → broadcast

```
Frame 42:
  t=0ms   BG packet arrives      → canvas created, timer armed if needed
  t=3ms   ROI tile 0 (person A)  → painted at (50, 120)
  t=5ms   ROI tile 1 (person B)  → painted at (380, 95)
  t=5ms   ROI count reached (2)  → flush immediately, broadcast
```

### Browser Rendering

The browser uses `createImageBitmap` instead of `new Image()`:

```
Traditional approach:          SASP approach:
new Image()                    createImageBitmap(blob)
→ decode on main thread        → decode on GPU compositor thread
→ blocks JS                    → never blocks JS
→ GC pressure at 30fps         → explicit bm.close() = no GC spikes
→ may tear                     → synced to display vsync via RAF
```

---

## The Protocol Header

Every UDP packet sent by Python carries a **28-byte binary header**:

```
Offset  Bytes  Type      Field
──────  ─────  ────────  ─────────────────────────────────────────
  0      4     char[4]   Magic: "SASP"
  4      1     uint8     Version: always 1
  5      1     uint8     Type: 0=Background, 1=ROI
  6      4     uint32    FrameID (wraps at 2^32 ≈ 4 billion)
 10      2     uint16    SeqNum (chunk index within this stream)
 12      2     uint16    TotalParts (total chunks for this payload)
 14      8     uint64    Timestamp (nanoseconds, time.time_ns())
 22      1     uint8     ROICount (total ROI tiles expected this frame)
 23      1     uint8     ROIIndex (which tile this is, 0-based)
 24      2     uint16    X (canvas X coordinate of ROI top-left)
 26      2     uint16    Y (canvas Y coordinate of ROI top-left)
──────  ─────  ────────  ─────────────────────────────────────────
Total: 28 bytes
```

**Why UDP and not TCP?** A dropped UDP frame means one glitched frame — barely visible at 30fps. A dropped TCP packet means the entire stream stalls waiting for retransmission. For live video, jitter is always preferable to latency spikes.

**Why chunked?** Most networks have an MTU (Maximum Transmission Unit) of ~1500 bytes. A single PNG tile can be 15–50KB. We split it into 1400-byte chunks (with 28-byte headers), send them individually, and reassemble on the Go side.

---

## Metrics Explained

There are **two completely separate metric systems** running simultaneously:

### Client-side metrics (JavaScript)

Computed entirely in the browser from raw WebSocket data:

```javascript
// Every frame that arrives:
ws.onmessage = async (ev) => {
    rxFrames++;
    rxBytes += ev.data.size;           // actual bytes received
    const t0 = performance.now();
    const bitmap = await createImageBitmap(ev.data);
    latSamples.push(performance.now() - t0);  // decode time ≈ network latency proxy
};

// Every 1 second:
fps = rxFrames / elapsed              // e.g. 29.4
bw  = rxBytes  / elapsed / 1024       // e.g. 23.1 KB/s
lat = average(latSamples)             // e.g. 8 ms
```

These measure **what the browser actually received** — affected by WebSocket drops and rendering speed.

### Server-side metrics (Go `/stats`)

Computed in Go using lock-free atomic counters:

```go
// Every time a frame is broadcast:
atomic.AddUint64(&s.FramesOut, 1)
atomic.AddUint64(&s.BytesOut,  uint64(frameSize))

// /stats handler computes delta over last interval:
avgFPS = float64(framesOut - lastFramesOut) / elapsed
```

Exposed as JSON at `GET /stats`, polled by the browser every 1 second.

### Why both?

| Situation | Client FPS | Server FPS | Meaning |
|-----------|-----------|-----------|---------|
| Healthy | ~30 | ~30 | Everything working |
| Slow browser | ~15 | ~30 | Client can't decode fast enough |
| Network congestion | ~20 | ~30 | Frames lost between Go → browser |
| Camera stall | ~0 | ~0 | Python camera read failed |

---

## Troubleshooting

### "Cannot open camera"
```bash
# Check available cameras
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]"
# Set CAMERA_INDEX in main.py to the correct number
```

### Stream not appearing in browser
1. Check Terminal 1 (Go) is running and shows "Listening on :8080"
2. Check Terminal 2 (Python) is running and printing FPS stats
3. Open browser console (F12) and check for WebSocket errors
4. Make sure nothing else is using ports 5000 or 8080

### Very low FPS (< 15)
- YOLO inference is slow on CPU. Set `DETECTION_CONFIDENCE = 0.5` (fewer detections)
- Reduce resolution: set `FRAME_WIDTH = 416`, `FRAME_HEIGHT = 320`
- Use a smaller model: `yolo11n-seg.pt` (nano) instead of `yolo11s-seg.pt` (small)
- If you have a GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### High bandwidth (not seeing savings)
- Multiple people in frame → more ROI tiles → more data. This is expected.
- Check `BG_JPEG_QUALITY` is set to `10` (not higher) in `main.py`
- The savings are most visible with 1–2 people in a large scene

### "Module not found: scripts.detector"
Make sure you run Python from the `edge/` directory:
```bash
cd SASP_Project/edge
python main.py   # not python edge/main.py
```

### Traditional backend port conflict
If port 8081 is busy:
```bash
# Find what's using it
lsof -i :8081
# Or change the port in traditional_backend.go:
HTTPAddr = ":8082"
```

---

## Quick Reference — All URLs

| URL | What |
|-----|------|
| `http://localhost:8080` | SASP live stream + metrics |
| `http://localhost:8080/stats` | Raw JSON stats from Go |
| `http://localhost:8080/comparison.html` | Animated bandwidth comparison demo |
| `http://localhost:8081` | Traditional stream + metrics |
| `http://localhost:8081/stats` | Traditional raw JSON stats |

## Quick Reference — All Ports

| Port | Protocol | Direction | Purpose |
|------|---------|-----------|---------|
| 5000 | UDP | Python → Go | SASP semantic stream |
| 5001 | UDP | Python → Go | Traditional full-frame stream |
| 8080 | TCP/HTTP | Go → Browser | SASP WebSocket + HTTP server |
| 8081 | TCP/HTTP | Go → Browser | Traditional WebSocket + HTTP server |

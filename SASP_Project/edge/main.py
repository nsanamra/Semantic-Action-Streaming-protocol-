"""
main.py — SASP Edge Transmitter  (v3 — 5-star)

What's new vs v2:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. Producer-consumer pipeline: camera grab runs on main thread,    │
  │    YOLO inference + encode + transmit run on a dedicated worker    │
  │    thread via a queue. Camera never stalls waiting for YOLO.       │
  │                                                                     │
  │ 2. ROI count in protocol header: fClass now carries the exact      │
  │    number of ROI tiles for this frame (0 = no people, N = N        │
  │    people). Go uses this to know exactly when to flush the canvas  │
  │    instead of relying on a timeout window. Eliminates the 12ms     │
  │    collect-window latency tax when person count is known.          │
  │                                                                     │
  │ 3. Parallel ROI encoding: when multiple people are detected, their │
  │    PNG tiles are encoded concurrently via ThreadPoolExecutor.      │
  │    On a 4-core machine with 3 people: ~3× faster encode step.     │
  │                                                                     │
  │ 4. ROI sub-index in header: each ROI packet carries a roi_index   │
  │    field (0-based) in the priority byte so Go can track exactly    │
  │    which tiles have arrived without ambiguity.                     │
  │                                                                     │
  │ 5. Console telemetry: prints FPS, inference time, encode time,    │
  │    bytes/frame every second so you can profile in real time.      │
  └─────────────────────────────────────────────────────────────────────┘

Protocol header (28 bytes, big-endian):
    Offset  Bytes  Field
     0       4     Magic "SASP"
     4       1     Version  (always 1)
     5       1     Type     (0=BG, 1=ROI)
     6       4     FrameID  (uint32, wraps at 2^32)
    10       2     SeqNum   (chunk index within this packet stream)
    12       2     TotalParts
    14       8     Timestamp (ns)
    22       1     ROICount  (total ROI tiles for this frame, 0 if BG-only)
    23       1     ROIIndex  (which tile this is, 0-based; 0 for BG packets)
    24       2     X         (canvas X of ROI top-left)
    26       2     Y         (canvas Y of ROI top-left)
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import cv2
import socket
import struct

from scripts.detector import SemanticDetector

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_IP    = "127.0.0.1"
SERVER_PORT  = 5000
CAMERA_INDEX = 0        # int for /dev/videoN, or RTSP URL string

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30
FRAME_BUDGET = 1.0 / TARGET_FPS

MTU_SIZE     = 1400
FRAME_ID_MAX = 2 ** 32

MAGIC   = b"SASP"
VERSION = 1

TYPE_BACKGROUND = 0
TYPE_ROI        = 1

# Worker queue depth — if the worker falls >2 frames behind, drop the oldest
PIPELINE_QUEUE_DEPTH = 2

# PNG compression level for ROI tiles (0=none,1=fastest…9=smallest)
PNG_COMPRESSION = 1

# JPEG quality for background (lower = smaller = faster to transmit)
BG_JPEG_QUALITY = 22
TRAD_JPEG_QUALITY = 80

# Adaptive Streaming Settings
MODE_TRADITIONAL = "traditional"
MODE_SASP        = "sasp"

LATENCY_THRESH_HIGH = 100.0   # p50 ms — switch to SASP immediately
LATENCY_THRESH_LOW  = 40.0    # p50 ms — prepare to switch to Traditional
UPGRADE_WAIT_SEC    = 3       # Seconds of stability before upgrading

# Global state updated by the background poller
g_streaming_mode = MODE_SASP
g_upgrade_counter = 0

# ─────────────────────────────────────────────────────────────────────────────
#  Adaptive Bandwidth Poller
# ─────────────────────────────────────────────────────────────────────────────

def _poll_metrics(stop_event: threading.Event) -> None:
    global g_streaming_mode
    while not stop_event.is_set():
        try:
            req = urllib.request.Request("http://localhost:8080/api/metrics")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                payload = json.loads(response.read().decode())
                data = payload.get("data", {})
                
                force_mode = data.get("force_mode", "auto")
                lat_p50    = data.get("latency_p50_ms", 0.0)
                drops      = data.get("dropped_frames", 0)

                if force_mode == "traditional":
                    g_streaming_mode = MODE_TRADITIONAL
                elif force_mode == "sasp":
                    g_streaming_mode = MODE_SASP
                else:
                    # Auto mode based on network health (Latency and Dropped frames)
                    if lat_p50 > LATENCY_THRESH_HIGH or drops > 0:
                        # Congestion detected -> Drop down to SASP immediately
                        if g_streaming_mode == MODE_TRADITIONAL:
                            g_streaming_mode = MODE_SASP
                            print(f"[adaptive] Congestion! Latency={lat_p50:.1f}ms Drops={drops}. Switching to SASP.")
                        g_upgrade_counter = 0

                    elif lat_p50 < LATENCY_THRESH_LOW and drops == 0:
                        # Network is crystal clear -> Prepare to upgrade
                        if g_streaming_mode == MODE_SASP:
                            g_upgrade_counter += 1
                            if g_upgrade_counter >= UPGRADE_WAIT_SEC:
                                g_streaming_mode = MODE_TRADITIONAL
                                g_upgrade_counter = 0
                                print(f"[adaptive] Network healthy for {UPGRADE_WAIT_SEC}s. Upgrading to TRADITIONAL.")
                    else:
                        if g_streaming_mode == MODE_SASP:
                            g_upgrade_counter = 0
        except Exception:
            pass  # Server might be restarting, just keep current mode
        
        # Poll every 1 second
        stop_event.wait(1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Telemetry  (printed to console every second)
# ─────────────────────────────────────────────────────────────────────────────

class Telemetry:
    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._frames     = 0
        self._infer_ms   = deque(maxlen=60)
        self._encode_ms  = deque(maxlen=60)
        self._bytes_sent = 0
        self._persons    = deque(maxlen=60)
        self._start      = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def record(self, *, infer_ms: float, encode_ms: float,
               bytes_sent: int, persons: int) -> None:
        with self._lock:
            self._frames     += 1
            self._infer_ms.append(infer_ms)
            self._encode_ms.append(encode_ms)
            self._bytes_sent += bytes_sent
            self._persons.append(persons)

    def _loop(self) -> None:
        while True:
            time.sleep(1.0)
            with self._lock:
                elapsed = time.time() - self._start
                fps   = self._frames / elapsed if elapsed > 0 else 0
                infer = sum(self._infer_ms) / len(self._infer_ms) if self._infer_ms else 0
                enc   = sum(self._encode_ms) / len(self._encode_ms) if self._encode_ms else 0
                bw    = self._bytes_sent / elapsed / 1024 if elapsed > 0 else 0
                ppl   = sum(self._persons) / len(self._persons) if self._persons else 0
                self._frames     = 0
                self._bytes_sent = 0
                self._start      = time.time()
            print(
                f"[sasp] fps={fps:4.1f}  "
                f"infer={infer:5.1f}ms  "
                f"encode={enc:5.1f}ms  "
                f"bw={bw:6.1f}KB/s  "
                f"persons={ppl:.1f}  "
                f"mode={g_streaming_mode.upper()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Transmitter
# ─────────────────────────────────────────────────────────────────────────────

class SASPTransmitter:
    def __init__(self, server_ip: str = SERVER_IP, server_port: int = SERVER_PORT):
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.server_addr = (server_ip, server_port)
        self.frame_id    = 0

    def _pack_header(
        self,
        frame_type: int,
        seq_num:    int,
        total_parts: int,
        roi_count:  int = 0,   # total ROI tiles for this frame
        roi_index:  int = 0,   # which tile this packet belongs to
        x: int = 0,
        y: int = 0,
    ) -> bytes:
        return struct.pack(
            "!4sBB IHH Q BB HH",
            MAGIC, VERSION, frame_type,
            self.frame_id,
            seq_num, total_parts,
            time.time_ns(),
            roi_count & 0xFF,   # byte: 0-255 ROI tiles is plenty
            roi_index & 0xFF,
            max(0, int(x)),
            max(0, int(y)),
        )

    def send(
        self,
        data:       bytes,
        frame_type: int,
        roi_count:  int = 0,
        roi_index:  int = 0,
        x: int = 0,
        y: int = 0,
    ) -> int:
        """Send chunked data; returns total bytes sent (headers + payload)."""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        sent = 0
        for i in range(total_parts):
            chunk  = data[i * MTU_SIZE : (i + 1) * MTU_SIZE]
            header = self._pack_header(
                frame_type, i, total_parts,
                roi_count=roi_count, roi_index=roi_index, x=x, y=y,
            )
            self.sock.sendto(header + chunk, self.server_addr)
            sent += len(header) + len(chunk)
        return sent

    def close(self) -> None:
        self.sock.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Worker  (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_roi(args: tuple) -> tuple[bytes, int, int]:
    """Encode one ROI tile. Returns (bytes, x, y).
    
    To solve the massive bandwidth problem of lossless PNG, we split the ROI:
      - RGB channels are compressed as High Quality JPEG
      - Alpha channel is compressed circularly as a grayscale PNG
    The payload is packed as: [4-byte LittleEndian uint32 JPG_LEN] [JPG_BYTES] [PNG_BYTES]
    """
    roi_rgba, fx1, fy1 = args
    
    roi_rgb = roi_rgba[:, :, :3]
    roi_alpha = roi_rgba[:, :, 3]

    # Compress RGB as JPEG (Quality 85 is visually flawless here)
    _, jpg_buf = cv2.imencode('.jpg', roi_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Compress Alpha mask (Since it's highly homogeneous, PNG compression collapses it to ~1-3KB)
    _, png_buf = cv2.imencode('.png', roi_alpha, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    jpg_bytes = jpg_buf.tobytes()
    png_bytes = png_buf.tobytes()
    
    # Pack into a custom compound payload for the Go server
    payload = struct.pack("<I", len(jpg_bytes)) + jpg_bytes + png_bytes
    
    return payload, fx1, fy1


def _worker(
    frame_queue: queue.Queue,
    detector:    SemanticDetector,
    transmitter: SASPTransmitter,
    telemetry:   Telemetry,
    stop_event:  threading.Event,
) -> None:
    pool = ThreadPoolExecutor(max_workers=4)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        bytes_sent = 0
        current_mode = g_streaming_mode

        if current_mode == MODE_TRADITIONAL:
            # ── Traditional Mode: Skip YOLO, Skip blur ────────────────────────
            t0 = time.perf_counter()
            infer_ms = 0.0
            roi_count = 0
            
            t1 = time.perf_counter()
            _, bg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, TRAD_JPEG_QUALITY])
            bg_bytes = bg_buf.tobytes()

            bytes_sent += transmitter.send(
                bg_bytes, TYPE_BACKGROUND,
                roi_count=0,
                roi_index=0,
            )
            encode_ms = (time.perf_counter() - t1) * 1000

        else:
            # ── SASP Mode: YOLO + Blur + Composite ────────────────────────────
            t0 = time.perf_counter()
            detections  = detector.detect(frame)
            roi_count   = detector.last_roi_count
            infer_ms    = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()

            if detections:
                encode_args = [
                    (det['roi_rgba'], det['bbox'][0], det['bbox'][1])
                    for det in detections
                ]
                encoded = list(pool.map(_encode_roi, encode_args))

                for idx, (roi_bytes, fx1, fy1) in enumerate(encoded):
                    bytes_sent += transmitter.send(
                        roi_bytes, TYPE_ROI,
                        roi_count=roi_count,
                        roi_index=idx,
                        x=fx1, y=fy1,
                    )

            blurred_bg  = detector.get_background(frame)
            _, bg_buf   = cv2.imencode('.jpg', blurred_bg, [cv2.IMWRITE_JPEG_QUALITY, BG_JPEG_QUALITY])
            bg_bytes    = bg_buf.tobytes()

            bytes_sent += transmitter.send(
                bg_bytes, TYPE_BACKGROUND,
                roi_count=roi_count,
                roi_index=0,
            )

            encode_ms = (time.perf_counter() - t1) * 1000

        # ── Advance frame ID ─────────────────────────────────────────────────
        transmitter.frame_id = (transmitter.frame_id + 1) % FRAME_ID_MAX

        telemetry.record(
            infer_ms=infer_ms,
            encode_ms=encode_ms,
            bytes_sent=bytes_sent,
            persons=roi_count,
        )

    pool.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_sasp_edge() -> None:
    detector    = SemanticDetector()
    transmitter = SASPTransmitter()
    telemetry   = Telemetry()
    stop_event  = threading.Event()

    frame_queue: queue.Queue = queue.Queue(maxsize=PIPELINE_QUEUE_DEPTH)

    worker_thread = threading.Thread(
        target=_worker,
        args=(frame_queue, detector, transmitter, telemetry, stop_event),
        daemon=True,
        name="sasp-worker",
    )

    poller_thread = threading.Thread(
        target=_poll_metrics,
        args=(stop_event,),
        daemon=True,
        name="sasp-poller",
    )

    # ── Camera setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[sasp] ERROR: Could not open camera.")
        sys.exit(1)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(sig, _frame):
        print("\n[sasp] Shutting down…")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("══════════════════════════════════════════════════════")
    print("  SASP Edge Transmitter  v3  —  Pipelined + Per-Person")
    print(f"  → {SERVER_IP}:{SERVER_PORT}   target {TARGET_FPS} FPS")
    print("══════════════════════════════════════════════════════")

    poller_thread.start()
    worker_thread.start()

    # ── Camera grab loop (main thread — never stalls) ────────────────────────
    while not stop_event.is_set() and cap.isOpened():
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[sasp] Camera read failed — exiting.")
            stop_event.set()
            break

        # Drop oldest frame if worker is backed up (non-blocking put)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put_nowait(frame)

        # Throttle grab loop to TARGET_FPS
        elapsed    = time.perf_counter() - loop_start
        sleep_for  = max(0.0, FRAME_BUDGET - elapsed)
        if sleep_for:
            time.sleep(sleep_for)

    cap.release()
    stop_event.set()
    worker_thread.join(timeout=3.0)
    transmitter.close()
    print("[sasp] Transmitter stopped.")


if __name__ == "__main__":
    run_sasp_edge()
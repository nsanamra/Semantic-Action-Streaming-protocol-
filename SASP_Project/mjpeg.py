"""
mjpeg_server.py — Industry-Standard MJPEG Baseline Server

This is the REAL traditional benchmark — not SASP's "traditional mode".

What this does:
  • Opens the same camera at the same resolution as SASP (640×480 @ 30fps)
  • Encodes every frame as a full JPEG (no segmentation, no blurring, no ROI)
  • Streams over HTTP multipart/x-mixed-replace — the actual industry standard
    used by IP cameras, OBS, VLC, surveillance systems, and browser-based feeds
  • Exposes /metrics JSON so the comparison dashboard can read bandwidth & FPS
  • Runs on port 8081 so it doesn't touch SASP on 8080

Ports:
    http://localhost:8081/stream   ← MJPEG stream (drop into <img src=>)
    http://localhost:8081/metrics  ← JSON: fps, bandwidth_kbps, bytes_per_frame
    http://localhost:8081/         ← Simple test page

Run alongside SASP:
    python mjpeg_server.py &
    python main.py
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  — match these to SASP's settings
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX   = 0      # same camera as SASP
FRAME_WIDTH    = 640    # same resolution as SASP
FRAME_HEIGHT   = 480
TARGET_FPS     = 30     # same target FPS as SASP

HTTP_PORT      = 8081   # SASP is on 8080 — we stay out of its way

# JPEG quality for MJPEG stream
# 80 = what real IP cameras typically ship with (Axis, Hikvision default)
# Try 22 too (SASP BG quality) to show quality vs bandwidth tradeoff
MJPEG_QUALITY  = 80

# ─────────────────────────────────────────────────────────────────────────────
#  Shared State  (written by capture thread, read by HTTP handlers)
# ─────────────────────────────────────────────────────────────────────────────

class SharedFrame:
    """Lock-protected latest JPEG frame + metrics ring buffer."""

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._jpeg: bytes = b""

        # Metrics (ring buffers, last 60 samples ≈ 2 seconds at 30fps)
        self._frame_times:  deque[float] = deque(maxlen=60)
        self._frame_sizes:  deque[int]   = deque(maxlen=60)
        self._total_bytes:  int          = 0
        self._total_frames: int          = 0
        self._start_time:   float        = time.time()

    def put(self, jpeg: bytes) -> None:
        now = time.time()
        with self._lock:
            self._jpeg = jpeg
            self._frame_times.append(now)
            self._frame_sizes.append(len(jpeg))
            self._total_bytes  += len(jpeg)
            self._total_frames += 1

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def get_metrics(self) -> dict:
        with self._lock:
            n = len(self._frame_times)
            if n < 2:
                fps = 0.0
                bw  = 0.0
                avg_bytes = 0
            else:
                # FPS from the ring buffer window
                window_sec = self._frame_times[-1] - self._frame_times[0]
                fps = (n - 1) / window_sec if window_sec > 0 else 0.0

                # Bandwidth: bytes sent in that window
                bw_bytes_per_sec = sum(self._frame_sizes) / window_sec if window_sec > 0 else 0.0
                bw = bw_bytes_per_sec / 1024.0  # KB/s

                avg_bytes = int(sum(self._frame_sizes) / n)

            return {
                "fps":               round(fps, 2),
                "bandwidth_kbps":    round(bw, 2),
                "bytes_per_frame":   avg_bytes,
                "total_frames":      self._total_frames,
                "total_mb_sent":     round(self._total_bytes / (1024 * 1024), 2),
                "uptime_seconds":    round(time.time() - self._start_time, 1),
                "jpeg_quality":      MJPEG_QUALITY,
                "resolution":        f"{FRAME_WIDTH}×{FRAME_HEIGHT}",
                "protocol":          "MJPEG/HTTP multipart",
            }


shared = SharedFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  Capture Thread  — grabs camera, encodes JPEG, updates shared state
# ─────────────────────────────────────────────────────────────────────────────

def capture_loop(stop_event: threading.Event) -> None:
    #cap = cv2.VideoCapture(CAMERA_INDEX)
    cap = cv2.VideoCapture("./edge/single_person.mp4")
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[mjpeg] ERROR: Could not open camera. Is SASP already using it?")
        print("[mjpeg] TIP: If SASP holds the camera exclusively, run this first,")
        print("         or use an RTSP source / virtual camera (v4l2loopback).")
        stop_event.set()
        return

    frame_budget = 1.0 / TARGET_FPS
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]

    print(f"[mjpeg] Capture started — {FRAME_WIDTH}×{FRAME_HEIGHT} @ {TARGET_FPS}fps  quality={MJPEG_QUALITY}")

    while not stop_event.is_set():
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[mjpeg] Camera read failed — stopping capture.")
            stop_event.set()
            break

        # Pure JPEG encode — no YOLO, no blur, no segmentation
        # This is exactly what a traditional IP camera firmware does
        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if ok:
            shared.put(buf.tobytes())

        elapsed   = time.perf_counter() - t0
        sleep_for = max(0.0, frame_budget - elapsed)
        if sleep_for:
            time.sleep(sleep_for)

    cap.release()
    print("[mjpeg] Capture stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP Handler
# ─────────────────────────────────────────────────────────────────────────────

BOUNDARY = b"--mjpegboundary"

class MJPEGHandler(BaseHTTPRequestHandler):
    """Handles three routes:
       /         → HTML test page
       /stream   → MJPEG multipart stream
       /metrics  → JSON metrics (polled by comparison dashboard)
    """

    # Silence the default per-request access log (noisy at 30fps)
    def log_message(self, fmt, *args):
        if self.path not in ("/stream",):
            print(f"[mjpeg] {self.address_string()} {fmt % args}")

    def handle_error(self):
        pass  # suppress BrokenPipeError tracebacks — normal on Mac/Safari

    def do_GET(self):
        if self.path == "/stream":
            self._serve_stream()
        elif self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        elif self.path in ("/comparison", "/comparison.html", "/"):
            self._serve_comparison()
        else:
            self._serve_index()

    def _serve_comparison(self):
        import os
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison.html")
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._serve_index()

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary=mjpegboundary")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                jpeg = shared.get_jpeg()
                if not jpeg:
                    time.sleep(0.01)
                    continue

                part = (
                    BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" +
                    jpeg +
                    b"\r\n"
                )
                self.wfile.write(part)
                self.wfile.flush()

                # Throttle to TARGET_FPS — don't hammer the browser
                time.sleep(1.0 / TARGET_FPS)

        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected — normal

    def _serve_metrics(self):
        try:
            data = shared.get_metrics()
            body = json.dumps({"status": "ok", "data": data, "timestamp": int(time.time() * 1000)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_health(self):
        try:
            body = json.dumps({"healthy": True, "protocol": "MJPEG/HTTP"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_index(self):
        html = f"""<!DOCTYPE html>
<html><head><title>MJPEG Baseline</title>
<style>body{{background:#111;color:#eee;font-family:monospace;text-align:center;padding:2rem}}
img{{max-width:100%;border:2px solid #444;border-radius:4px}}
pre{{background:#1a1a1a;padding:1rem;border-radius:4px;text-align:left;display:inline-block}}</style>
</head><body>
<h2>MJPEG Baseline Server</h2>
<p>Quality: {MJPEG_QUALITY} | Resolution: {FRAME_WIDTH}×{FRAME_HEIGHT} | Protocol: multipart/x-mixed-replace</p>
<img src="/stream" alt="MJPEG Stream"><br><br>
<pre>Stream URL : http://localhost:{HTTP_PORT}/stream
Metrics URL: http://localhost:{HTTP_PORT}/metrics</pre>
</body></html>""".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import signal, sys
    from http.server import ThreadingHTTPServer

    stop_event = threading.Event()

    # Capture thread
    cap_thread = threading.Thread(target=capture_loop, args=(stop_event,), daemon=True, name="mjpeg-capture")
    cap_thread.start()

    # ThreadingHTTPServer handles each request in its own thread —
    # critical so that one slow /stream client doesn't block /metrics
    class _QuietThreadingServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass  # swallow BrokenPipeError noise

    server = _QuietThreadingServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)

    print("══════════════════════════════════════════════════════")
    print("  MJPEG Baseline Server  —  Industry Standard")
    print(f"  Comparison → http://localhost:{HTTP_PORT}/comparison")
    print(f"  Stream     → http://localhost:{HTTP_PORT}/stream")
    print(f"  Metrics    → http://localhost:{HTTP_PORT}/metrics")
    print(f"  Quality={MJPEG_QUALITY}  Resolution={FRAME_WIDTH}×{FRAME_HEIGHT}  FPS={TARGET_FPS}")
    print("  Press Ctrl+C to stop.")
    print("══════════════════════════════════════════════════════")

    # Run serve_forever in a daemon thread so the main thread stays free
    # to catch SIGINT. On macOS, signal handlers only fire on the main thread,
    # so blocking it with serve_forever() prevents Ctrl+C from working.
    server_thread = threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-http")
    server_thread.start()

    try:
        # Main thread just waits — signals will interrupt this sleep loop
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[mjpeg] Shutting down…")
        stop_event.set()
        server.shutdown()       # signals serve_forever() to exit
        server_thread.join(timeout=3)
        print("[mjpeg] Done.")


if __name__ == "__main__":
    main()
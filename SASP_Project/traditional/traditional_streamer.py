"""
traditional_streamer.py — Naive full-frame JPEG streamer

This is the "dumb" baseline that most streaming systems use.
Every frame: capture → JPEG encode at high quality → send entire frame over UDP.
No semantic understanding. No ROI. No blurring. Just brute-force transmission.

Run this in a SEPARATE terminal:
    python traditional_streamer.py

It streams to port 5001 (the traditional Go backend listens there).
Open http://localhost:8081 to see it.
"""

import signal
import sys
import time
import socket
import struct
import threading
from collections import deque

import cv2

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

SERVER_IP    = "127.0.0.1"
SERVER_PORT  = 5001          # different port from SASP (5000)

CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30
FRAME_BUDGET = 1.0 / TARGET_FPS

JPEG_QUALITY = 80            # high quality — as a naive streamer would use
MTU_SIZE     = 1400
FRAME_ID_MAX = 2 ** 32

# Simple 8-byte header: FrameID(4) + SeqNum(2) + TotalParts(2)
# Intentionally minimal — traditional streamers don't need semantic headers
HEADER_FMT = "!IHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # = 8 bytes


# ─────────────────────────────────────────────────────────────────────────────
#  Telemetry
# ─────────────────────────────────────────────────────────────────────────────

class Telemetry:
    def __init__(self):
        self._lock       = threading.Lock()
        self._frames     = 0
        self._bytes      = 0
        self._encode_ms  = deque(maxlen=60)
        self._start      = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def record(self, bytes_sent: int, encode_ms: float):
        with self._lock:
            self._frames += 1
            self._bytes  += bytes_sent
            self._encode_ms.append(encode_ms)

    def _loop(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                elapsed  = time.time() - self._start
                fps      = self._frames / elapsed if elapsed > 0 else 0
                bw       = self._bytes  / elapsed / 1024 if elapsed > 0 else 0
                enc      = (sum(self._encode_ms)/len(self._encode_ms)
                            if self._encode_ms else 0)
                self._frames = 0
                self._bytes  = 0
                self._start  = time.time()
            print(
                f"[traditional] fps={fps:4.1f}  "
                f"encode={enc:5.1f}ms  "
                f"bw={bw:7.1f} KB/s  "
                f"(~{bw/1024:.2f} MB/s)"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Streamer
# ─────────────────────────────────────────────────────────────────────────────

class TraditionalStreamer:
    def __init__(self):
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (SERVER_IP, SERVER_PORT)
        self.frame_id    = 0

    def send_frame(self, data: bytes) -> int:
        """Chunk full-frame JPEG into MTU-sized UDP packets."""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        sent = 0
        for i in range(total_parts):
            chunk  = data[i * MTU_SIZE : (i + 1) * MTU_SIZE]
            header = struct.pack(HEADER_FMT, self.frame_id, i, total_parts)
            self.sock.sendto(header + chunk, self.server_addr)
            sent += HEADER_SIZE + len(chunk)
        return sent

    def close(self):
        self.sock.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run():
    streamer  = TraditionalStreamer()
    telemetry = Telemetry()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  FRAME_HEIGHT)

    if not cap.isOpened():
        print("[traditional] ERROR: Cannot open camera.")
        sys.exit(1)

    running = True
    def _shutdown(sig, _):
        nonlocal running
        print("\n[traditional] Shutting down…")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("══════════════════════════════════════════════════")
    print("  Traditional Full-Frame Streamer")
    print(f"  → {SERVER_IP}:{SERVER_PORT}   JPEG quality={JPEG_QUALITY}%")
    print(f"  View at: http://localhost:8081")
    print("══════════════════════════════════════════════════")

    while running and cap.isOpened():
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[traditional] Camera read failed.")
            break

        # Encode entire frame at high quality — no blurring, no segmentation
        t0 = time.perf_counter()
        _, buf = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        encode_ms  = (time.perf_counter() - t0) * 1000
        frame_data = buf.tobytes()

        bytes_sent = streamer.send_frame(frame_data)
        telemetry.record(bytes_sent, encode_ms)

        streamer.frame_id = (streamer.frame_id + 1) % FRAME_ID_MAX

        elapsed   = time.perf_counter() - loop_start
        sleep_for = max(0.0, FRAME_BUDGET - elapsed)
        if sleep_for:
            time.sleep(sleep_for)

    cap.release()
    streamer.close()
    print("[traditional] Stopped.")


if __name__ == "__main__":
    run()
"""
traditional_streamer.py — Simple full-frame webcam streamer

No YOLO. No segmentation. No blurring.
Just: grab frame → JPEG encode → send over UDP → repeat.

This is the baseline to compare against SASP.

Usage:
    python traditional_streamer.py

Streams to 127.0.0.1:5001
View at http://localhost:8081
"""

import cv2
import socket
import struct
import time
import signal
import sys

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_IP   = "127.0.0.1"
SERVER_PORT = 5001

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 30
JPEG_QUALITY  = 80      # same quality a normal streaming app would use
MTU_SIZE      = 1400
FRAME_ID_MAX  = 2 ** 32

# Header: FrameID(4) + SeqNum(2) + TotalParts(2) + Timestamp_ns(8) = 16 bytes
HEADER_FMT  = "!IHH Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # = 16 bytes

# ── Streamer ──────────────────────────────────────────────────────────────────

class TraditionalStreamer:
    def __init__(self):
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.server_addr = (SERVER_IP, SERVER_PORT)
        self.frame_id    = 0

    def send_frame(self, data: bytes) -> int:
        """Chunk JPEG into MTU-sized UDP packets. Returns total bytes sent."""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        sent = 0
        ts = time.time_ns()
        for i in range(total_parts):
            chunk  = data[i * MTU_SIZE : (i + 1) * MTU_SIZE]
            header = struct.pack(HEADER_FMT, self.frame_id, i, total_parts, ts)
            self.sock.sendto(header + chunk, self.server_addr)
            sent += HEADER_SIZE + len(chunk)
        self.frame_id = (self.frame_id + 1) % FRAME_ID_MAX
        return sent

    def close(self):
        self.sock.close()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    streamer = TraditionalStreamer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[trad] ERROR: Cannot open camera.")
        sys.exit(1)

    running = True
    def _shutdown(sig, _):
        nonlocal running
        print("\n[trad] Shutting down...")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    frame_budget = 1.0 / TARGET_FPS

    # Telemetry counters
    tele_frames = 0
    tele_bytes  = 0
    tele_start  = time.time()

    print("=" * 52)
    print("  Traditional Streamer — NO segmentation, NO blurring")
    print(f"  → {SERVER_IP}:{SERVER_PORT}   JPEG quality={JPEG_QUALITY}%")
    print(f"  View at: http://localhost:8081")
    print("=" * 52)

    while running and cap.isOpened():
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[trad] Camera read failed.")
            break

        _, buf  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        sent    = streamer.send_frame(buf.tobytes())

        tele_frames += 1
        tele_bytes  += sent

        # Print telemetry every second
        elapsed = time.time() - tele_start
        if elapsed >= 1.0:
            fps = tele_frames / elapsed
            bw  = tele_bytes  / elapsed / 1024
            print(f"[trad] fps={fps:4.1f}  bw={bw:7.1f} KB/s  "
                  f"frame_size={tele_bytes//max(tele_frames,1)//1024:.1f} KB")
            tele_frames = 0
            tele_bytes  = 0
            tele_start  = time.time()

        sleep_for = max(0.0, frame_budget - (time.perf_counter() - loop_start))
        if sleep_for:
            time.sleep(sleep_for)

    cap.release()
    streamer.close()
    print("[trad] Stopped.")


if __name__ == "__main__":
    run()
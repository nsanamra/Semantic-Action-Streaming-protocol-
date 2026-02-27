"""
transmitter.py — SASP Standalone Demo Transmitter

This is the self-contained demo / Phase-1 transmitter that doesn't require
a YOLO model. It uses a mock centre-crop ROI so you can test the full
pipeline end-to-end with nothing but a webcam.

Key fixes vs v1:
  • Header is now 28 bytes (added X, Y fields) — matches the Go backend.
  • frame_id wraps at 2^32 (uint32 parity with Go).
  • Graceful SIGINT / camera-failure shutdown.
  • Proper error logging instead of silent failures.
"""

import signal
import sys
import cv2
import socket
import struct
import time

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_IP   = "127.0.0.1"
SERVER_PORT = 5000

MTU_SIZE     = 1400
FRAME_ID_MAX = 2 ** 32

# SASP Protocol constants
MAGIC   = b"SASP"
VERSION = 1

TYPE_BACKGROUND = 0
TYPE_ROI        = 1


# ─────────────────────────────────────────────────────────────────────────────
#  Transmitter
# ─────────────────────────────────────────────────────────────────────────────

class SASPTransmitter:
    """
    28-byte SASP header (big-endian):
        Magic(4) Version(1) Type(1) FrameID(4) Seq(2) Total(2)
        Timestamp(8) Class(1) Priority(1) X(2) Y(2)
    """

    def __init__(self, server_ip: str = SERVER_IP, server_port: int = SERVER_PORT):
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (server_ip, server_port)
        self.frame_id    = 0

    def _pack_header(
        self,
        frame_type: int,
        seq_num: int,
        total_parts: int,
        obj_class: int = 0,
        priority: int  = 10,
        x: int         = 0,
        y: int         = 0,
    ) -> bytes:
        return struct.pack(
            "!4sBB IHH Q BB HH",
            MAGIC,
            VERSION,
            frame_type,
            self.frame_id,
            seq_num,
            total_parts,
            time.time_ns(),
            obj_class,
            priority,
            max(0, int(x)),
            max(0, int(y)),
        )

    def send(
        self,
        data: bytes,
        frame_type: int,
        priority: int,
        x: int = 0,
        y: int = 0,
        obj_class: int = 0,
    ) -> None:
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        for i in range(total_parts):
            chunk  = data[i * MTU_SIZE : (i + 1) * MTU_SIZE]
            header = self._pack_header(
                frame_type, i, total_parts,
                obj_class=obj_class, priority=priority, x=x, y=y,
            )
            try:
                self.sock.sendto(header + chunk, self.server_addr)
            except OSError as e:
                print(f"[transmitter] send error: {e}")
            # if frame_type == TYPE_BACKGROUND:
            #     time.sleep(0.0005)

    def close(self) -> None:
        self.sock.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Demo loop
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    transmitter = SASPTransmitter()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[transmitter] ERROR: Could not open camera.")
        sys.exit(1)

    running = True

    def _shutdown(sig, frame):
        nonlocal running
        print("\n[transmitter] Shutting down…")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[transmitter] Demo mode → {SERVER_IP}:{SERVER_PORT}")
    print("[transmitter] Using mock centre-crop ROI (no YOLO required)")

    while running and cap.isOpened():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[transmitter] Camera read failed.")
            break

        h, w = frame.shape[:2]

        # ── Mock ROI: centre 20 % of the frame ───────────────────────────────
        rw, rh = int(w * 0.2), int(h * 0.2)
        rx,  ry = int(w * 0.4), int(h * 0.4)
        roi_frame = frame[ry:ry+rh, rx:rx+rw]

        # ── Background: blurred, low quality ─────────────────────────────────
        blurred = cv2.GaussianBlur(frame, (31, 31), 0)
        _, bg_enc = cv2.imencode('.jpg', blurred, [cv2.IMWRITE_JPEG_QUALITY, 10])

        transmitter.send(bg_enc.tobytes(), TYPE_BACKGROUND, priority=10, obj_class=1)

        # ── ROI: high quality JPEG (demo — no alpha) ──────────────────────────
        _, roi_enc = cv2.imencode('.jpg', roi_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        transmitter.send(
            roi_enc.tobytes(), TYPE_ROI,
            priority=200, x=rx, y=ry, obj_class=1,
        )

        transmitter.frame_id = (transmitter.frame_id + 1) % FRAME_ID_MAX

        # ── 30 fps throttle ───────────────────────────────────────────────────
        elapsed = time.time() - frame_start
        sleep_for = max(0.0, 0.033 - elapsed)
        if sleep_for:
            time.sleep(sleep_for)

    cap.release()
    transmitter.close()
    print("[transmitter] Stopped.")


if __name__ == "__main__":
    run()
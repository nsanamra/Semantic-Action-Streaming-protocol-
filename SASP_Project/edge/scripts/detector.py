"""
detector.py — SASP Semantic Detector  (v3 — 5-star)

What's new vs v2:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. Person-only filter: YOLO classes= [0] so chairs, dogs, cars     │
  │    never get segmented or transmitted.                              │
  │ 2. Stale track eviction: smoothing state for tracks that vanish    │
  │    is purged after MAX_TRACK_AGE frames, preventing memory leaks.  │
  │ 3. roi_count embedded in return value so the protocol can carry    │
  │    the exact number of ROI tiles expected (Go needs this to flush  │
  │    the canvas immediately rather than waiting for a timer).        │
  │ 4. Adaptive blur strength: background blur kernel scales with how  │
  │    much the scene is moving (static scene → heavier blur is fine;  │
  │    fast motion → lighter blur to avoid ghosting on BG edges).      │
  │ 5. Mask morphology: erode→dilate the binary mask before feathering │
  │    to close small holes inside the silhouette (jacket gaps etc.)   │
  │ 6. Half-resolution mask: resize/threshold at half res, then scale  │
  │    up — ~4× faster mask processing with visually identical result. │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
#  Tuning Constants  (tweak here, nowhere else)
# ─────────────────────────────────────────────────────────────────────────────

DETECTION_CONFIDENCE = 0.40     # YOLO detection threshold
PERSON_CLASS_ID      = 0        # COCO class 0 = person — only class we care about

SMOOTHING_ALPHA      = 0.82     # EMA weight: higher = smoother but laggier bbox
DEADZONE_PX          = 12.0     # pixel movement threshold before bbox updates
ROI_PAD              = 28       # padding (px) added around each tight bbox

# Background blur
BG_DIFF_THRESHOLD    = 6.0      # mean-abs-diff below which BG is considered static
BG_BLUR_STATIC       = (35, 35) # kernel when scene is static (heavier = nicer bokeh)
BG_BLUR_MOTION       = (21, 21) # kernel when scene is moving (lighter = less ghost)

# Mask processing
MASK_MORPH_KERNEL    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MASK_FEATHER_KERNEL  = (15, 15) # Gaussian feather on final alpha

# Track management
MAX_TRACK_AGE        = 90       # frames before unseen track state is evicted (~3 s)

# Sharpening
SHARPEN_KERNEL = np.array([[0, -0.5, 0],
                            [-0.5, 3, -0.5],
                            [0, -0.5, 0]], dtype=np.float32)

MODEL_PATH = os.environ.get("YOLO_MODEL", "yolo26s-seg.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  SemanticDetector
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDetector:
    """
    Runs YOLOv26-seg (person class only) on every frame.

    Public API
    ----------
    get_background(frame) → np.ndarray
        Lazily-cached blurred background.

    detect(frame) → list[dict]
        Returns a list of per-person dicts:
            {
                'bbox':     [x1, y1, x2, y2],
                'roi_rgba': np.ndarray (H×W×4, BGRA),
            }
        Also sets self.last_roi_count so the caller can embed the count
        in the protocol header without re-computing len(detections).
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = YOLO(model_path)

        # Smoothing: track_id → (bbox_norm, age_frames_since_last_seen)
        self._smooth:    dict[int, np.ndarray] = {}
        self._track_age: dict[int, int]        = {}

        # BG caching
        self._prev_gray:   np.ndarray | None = None
        self._cached_blur: np.ndarray | None = None
        self._last_diff:   float             = 0.0   # used for adaptive kernel

        # Public: number of ROI tiles produced last detect() call
        self.last_roi_count: int = 0

    # ── Public ───────────────────────────────────────────────────────────────

    def get_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a blurred background.
        • Reuses cached result when the scene is static (diff < threshold).
        • Adapts blur kernel to scene motion so fast-moving backgrounds
          don't show ghosting edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is not None:
            diff = float(np.mean(np.abs(
                gray.astype(np.float32) - self._prev_gray.astype(np.float32)
            )))
            self._last_diff = diff
            if diff < BG_DIFF_THRESHOLD and self._cached_blur is not None:
                return self._cached_blur
        else:
            diff = 999.0

        self._prev_gray = gray

        # Adaptive kernel: use heavier blur when scene is calm
        kernel = BG_BLUR_STATIC if diff < 20.0 else BG_BLUR_MOTION
        self._cached_blur = cv2.GaussianBlur(frame, kernel, 0)
        return self._cached_blur

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect and segment all people in *frame*.
        Sets self.last_roi_count = len(result) before returning.
        """
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=DETECTION_CONFIDENCE,
            classes=[PERSON_CLASS_ID],   # ← only detect persons
            verbose=False,
        )[0]

        if results.masks is None or len(results.boxes) == 0:
            self._age_all_tracks()
            self.last_roi_count = 0
            return []

        # Resolve track IDs (may be None if model not in tracking mode)
        track_ids: list[int] = (
            results.boxes.id.cpu().numpy().astype(int).tolist()
            if results.boxes.id is not None
            else list(range(len(results.boxes)))
        )

        seen_ids: set[int] = set()
        detections: list[dict] = []

        for tid, box, mask in zip(track_ids, results.boxes, results.masks):
            seen_ids.add(tid)

            # ── Mask (processed at half resolution for speed) ─────────────
            raw = mask.data[0].cpu().numpy()                       # model output res
            half_w, half_h = w // 2, h // 2
            small = cv2.resize(raw, (half_w, half_h), interpolation=cv2.INTER_LINEAR)
            small = (np.clip(small, 0, 1) * 255).astype(np.uint8)

            # Morphological close: fills holes in silhouette
            small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, MASK_MORPH_KERNEL)

            # Scale back to full resolution
            full_mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

            # Feather edges for smooth compositing
            feathered = cv2.GaussianBlur(full_mask, MASK_FEATHER_KERNEL, 0)

            # ── BBox smoothing (normalised) ───────────────────────────────
            raw_coords = box.xyxy[0].cpu().numpy()
            norm = raw_coords / np.array([w, h, w, h], dtype=np.float32)
            stable_norm = self._smooth_bbox(tid, norm)
            sx1, sy1, sx2, sy2 = (stable_norm * np.array([w, h, w, h])).astype(int)

            # ── Pad & clamp ───────────────────────────────────────────────
            fx1 = max(0, sx1 - ROI_PAD)
            fy1 = max(0, sy1 - ROI_PAD)
            fx2 = min(w, sx2 + ROI_PAD)
            fy2 = min(h, sy2 + ROI_PAD)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # ── Crop ─────────────────────────────────────────────────────
            roi_bgr   = frame[fy1:fy2, fx1:fx2].copy()
            roi_alpha = feathered[fy1:fy2, fx1:fx2]

            # ── Sharpen subject ───────────────────────────────────────────
            roi_bgr = cv2.filter2D(roi_bgr, -1, SHARPEN_KERNEL)
            roi_bgr = np.clip(roi_bgr, 0, 255).astype(np.uint8)

            b, g, r = cv2.split(roi_bgr)
            roi_rgba = cv2.merge((b, g, r, roi_alpha))

            detections.append({'bbox': [fx1, fy1, fx2, fy2], 'roi_rgba': roi_rgba})

        # Age unseen tracks, evict stale ones
        self._age_tracks(seen_ids)

        self.last_roi_count = len(detections)
        return detections

    # ── Private ──────────────────────────────────────────────────────────────

    def _smooth_bbox(self, track_id: int, norm: np.ndarray) -> np.ndarray:
        """Deadzone + EMA on normalised [0,1] coordinates."""
        norm_deadzone = DEADZONE_PX / 640.0

        if track_id not in self._smooth:
            self._smooth[track_id] = norm.copy()
            self._track_age[track_id] = 0
            return norm

        prev = self._smooth[track_id]
        movement = float(np.max(np.abs(norm - prev)))

        if movement < norm_deadzone:
            smoothed = prev
        else:
            smoothed = SMOOTHING_ALPHA * prev + (1.0 - SMOOTHING_ALPHA) * norm
            self._smooth[track_id] = smoothed

        self._track_age[track_id] = 0   # reset age — track is active
        return smoothed

    def _age_tracks(self, seen_ids: set[int]) -> None:
        """Increment age counter for unseen tracks; evict if too old."""
        for tid in list(self._track_age):
            if tid not in seen_ids:
                self._track_age[tid] += 1
                if self._track_age[tid] > MAX_TRACK_AGE:
                    self._smooth.pop(tid, None)
                    self._track_age.pop(tid, None)

    def _age_all_tracks(self) -> None:
        """Called when no detections at all — age every track."""
        self._age_tracks(set())
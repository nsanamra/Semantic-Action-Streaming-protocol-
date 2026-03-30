import os
import time
import argparse
import sys

import cv2
import numpy as np

# Require scikit-image for accurate SSIM
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Please install scikit-image to run SSIM evaluation:")
    print("pip install scikit-image opencv-python ultralytics")
    sys.exit(1)

from detector import SemanticDetector

def compute_metrics(raw, compressed, bbox=None):
    """Computes SSIM and PSNR between two BGR images. 
    If bbox is provided (x1, y1, x2, y2), it computes metrics ONLY for that region."""
    
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(raw.shape[1], x2), min(raw.shape[0], y2)
        
        raw_region = raw[y1:y2, x1:x2]
        comp_region = compressed[y1:y2, x1:x2]
    else:
        raw_region = raw
        comp_region = compressed
        
    if raw_region.size == 0 or comp_region.size == 0:
        return 0.0, 0.0

    psnr_val = cv2.PSNR(raw_region, comp_region)
    
    # SSIM requires grayscale or multi-channel handling. Data range is 255 for 8-bit.
    ssim_val, _ = ssim(raw_region, comp_region, channel_axis=2, full=True, data_range=255)
    
    return float(ssim_val), float(psnr_val)

def main():
    parser = argparse.ArgumentParser(description="Generate SSIM Paper Data")
    parser.add_argument("--image", type=str, required=True, help="Path to a test image with a person in it")
    args = parser.parse_args()

    raw_frame = cv2.imread(args.image)
    if raw_frame is None:
        print(f"Could not load image {args.image}")
        return

    detector = SemanticDetector()
    
    print(f"--- Evaluating {args.image} ({raw_frame.shape[1]}x{raw_frame.shape[0]}) ---")
    
    # ── 1. Simulate Traditional Compression (e.g., 500 KB/s equivalent) ──
    # Heavily degrade the frame to simulate a choking network
    TRAD_QUALITY = 15
    _, trad_buf = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, TRAD_QUALITY])
    trad_compressed = cv2.imdecode(trad_buf, cv2.IMREAD_COLOR)
    trad_size_kb = len(trad_buf) / 1024.0

    # ── 2. Simulate SASP Semantic Compression ──
    t0 = time.perf_counter()
    detections = detector.detect(raw_frame)
    infer_time = time.perf_counter() - t0
    
    if not detections:
        print("No person detected by YOLO. Please provide an image with a person.")
        return
        
    # Get SASP background (blurred to save bandwidth)
    bg_blurred = detector.get_background(raw_frame)
    BG_QUALITY = 22 # Same as edge/main.py
    _, sasp_bg_buf = cv2.imencode('.jpg', bg_blurred, [cv2.IMWRITE_JPEG_QUALITY, BG_QUALITY])
    
    sasp_size_kb = len(sasp_bg_buf) / 1024.0
    
    # Reconstruct the SASP frame exactly how the Go server would composite it
    sasp_reconstructed = cv2.imdecode(sasp_bg_buf, cv2.IMREAD_COLOR)
    
    master_bbox = None
    for det in detections:
        bx1, by1, bx2, by2 = map(int, det['bbox'])
        roi_rgba = det['roi_rgba']
        
        # Simulate new Dual Payload (JPEG RGB + PNG Alpha)
        roi_rgb = roi_rgba[:, :, :3]
        roi_alpha = roi_rgba[:, :, 3]

        _, jpg_buf = cv2.imencode('.jpg', roi_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, mask_buf = cv2.imencode('.png', roi_alpha, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        sasp_size_kb += (len(jpg_buf) + len(mask_buf) + 4) / 1024.0
        
        # Composite ROI over background (Alpha blending)
        roi_rgb = roi_rgba[:, :, :3]
        roi_alpha = (roi_rgba[:, :, 3] / 255.0)[..., np.newaxis]
        
        bg_region = sasp_reconstructed[by1:by2, bx1:bx2]
        sasp_reconstructed[by1:by2, bx1:bx2] = (roi_alpha * roi_rgb + (1 - roi_alpha) * bg_region).astype(np.uint8)
        
        master_bbox = det['bbox'] # Evaluate the first person found

    print("\n[Size Overhead]")
    print(f"Traditional (Q={TRAD_QUALITY}): {trad_size_kb:.1f} KB")
    print(f"SASP (Q={BG_QUALITY} + PNG ROI): {sasp_size_kb:.1f} KB")
    print(f"YOLO Inference Latency: {infer_time*1000:.1f} ms")

    print("\n[Objective Quality: WHOLE FRAME]")
    trad_ssim_f, trad_psnr_f = compute_metrics(raw_frame, trad_compressed)
    sasp_ssim_f, sasp_psnr_f = compute_metrics(raw_frame, sasp_reconstructed)
    print(f"Traditional -> SSIM: {trad_ssim_f:.4f} | PSNR: {trad_psnr_f:.2f} dB")
    print(f"SASP        -> SSIM: {sasp_ssim_f:.4f} | PSNR: {sasp_psnr_f:.2f} dB")

    print("\n[Objective Quality: HIGH PRIORITY ROI (The Person)]")
    trad_ssim_roi, trad_psnr_roi = compute_metrics(raw_frame, trad_compressed, master_bbox)
    sasp_ssim_roi, sasp_psnr_roi = compute_metrics(raw_frame, sasp_reconstructed, master_bbox)
    print(f"Traditional -> SSIM: {trad_ssim_roi:.4f} | PSNR: {trad_psnr_roi:.2f} dB")
    print(f"SASP        -> SSIM: {sasp_ssim_roi:.4f} | PSNR: {sasp_psnr_roi:.2f} dB")
    
    print("\n(Use these ROI SSIM numbers in your Research Paper!)")

if __name__ == "__main__":
    main()

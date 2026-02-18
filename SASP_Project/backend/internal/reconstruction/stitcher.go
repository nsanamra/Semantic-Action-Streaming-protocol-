package reconstruction

import (
	"image"
	"image/draw"
)

// Stitch composites a transparent-alpha ROI tile onto a background image.
//
//	background  – the blurred full-frame JPEG decoded to image.Image
//	roi         – the sharp RGBA PNG containing the segmented subject(s)
//	x, y        – top-left canvas coordinates where the ROI should land
//
// The function is allocation-minimal: it creates one RGBA canvas, draws
// the background with draw.Src (no alpha blending needed), then overlays
// the ROI with draw.Over so the feathered alpha mask blends correctly.
func Stitch(background image.Image, roi image.Image, x, y int) image.Image {
	if background == nil {
		return roi
	}
	if roi == nil {
		return background
	}

	bounds := background.Bounds()
	result := image.NewRGBA(bounds)

	// 1. Blit the full blurred background
	draw.Draw(result, bounds, background, image.Point{}, draw.Src)

	// 2. Clip the destination rect to canvas bounds to prevent out-of-range
	//    draws (can happen when bbox smoothing places the subject near edges)
	dstRect := image.Rect(x, y, x+roi.Bounds().Dx(), y+roi.Bounds().Dy())
	dstRect = dstRect.Intersect(bounds)
	if dstRect.Empty() {
		return result
	}

	// 3. Overlay the ROI — draw.Over respects the PNG alpha channel,
	//    producing smooth, feathered edges from the Gaussian-blurred mask.
	draw.Draw(result, dstRect, roi, image.Point{}, draw.Over)

	return result
}

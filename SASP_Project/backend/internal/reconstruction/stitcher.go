package reconstruction

import (
	"image"
	"image/color"
	"image/draw"
)

// Stitch mathematically blends the high-quality ROI into the highly compressed background
func Stitch(background image.Image, roi image.Image, x, y int) image.Image {
	// Fallback if background packets were lost in transit
	if background == nil {
		return roi
	}

	bounds := background.Bounds()
	result := image.NewRGBA(bounds)

	// 1. Draw the blurred, 10% compressed background layer
	draw.Draw(result, bounds, background, image.Point{}, draw.Src)

	// 2. Setup the Feathering Mask
	roiBounds := roi.Bounds()
	mask := image.NewAlpha(roiBounds)

	// Massive 45-pixel radius to create a DSLR-like depth of field transition
	blendRadius := 45.0

	for i := 0; i < roiBounds.Dx(); i++ {
		for j := 0; j < roiBounds.Dy(); j++ {
			// Find shortest distance to the physical edge of the ROI box
			distX := minInt(i, roiBounds.Dx()-1-i)
			distY := minInt(j, roiBounds.Dy()-1-j)
			dist := float64(minInt(distX, distY))

			if dist < blendRadius {
				// Quadratic curve math: creates a non-linear, ultra-smooth transparent fade
				ratio := dist / blendRadius
				alpha := uint8((ratio * ratio) * 255)
				mask.SetAlpha(i, j, color.Alpha{A: alpha})
			} else {
				// The center mass of the target stays 100% opaque and sharp
				mask.SetAlpha(i, j, color.Alpha{A: 255})
			}
		}
	}

	// 3. Draw the ROI over the background utilizing the quadratic alpha mask
	dstRect := roiBounds.Add(image.Pt(x, y))
	draw.DrawMask(result, dstRect, roi, image.Point{}, mask, image.Point{}, draw.Over)

	return result
}

// Utility function to calculate the nearest edge
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

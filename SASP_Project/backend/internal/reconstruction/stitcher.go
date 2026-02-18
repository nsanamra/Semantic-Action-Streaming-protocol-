package reconstruction

import (
	"image"
	"image/color"
	"image/draw"
)

// Stitch blends the high-quality ROI into the compressed background using an alpha mask
func Stitch(background image.Image, roi image.Image, x, y int) image.Image {
	// If no background arrived yet, just return the ROI
	if background == nil {
		return roi
	}

	bounds := background.Bounds()
	result := image.NewRGBA(bounds)

	// 1. Draw the heavily compressed background first
	draw.Draw(result, bounds, background, image.Point{}, draw.Src)

	// 2. Create an Alpha Mask for the ROI to "feather" the edges
	roiBounds := roi.Bounds()
	mask := image.NewAlpha(roiBounds)

	// BlendRadius determines the width of the smooth gradient fade (pixels)
	blendRadius := 20.0

	for i := 0; i < roiBounds.Dx(); i++ {
		for j := 0; j < roiBounds.Dy(); j++ {
			// Calculate distance to nearest edge for gradient fade
			distX := minInt(i, roiBounds.Dx()-1-i)
			distY := minInt(j, roiBounds.Dy()-1-j)
			dist := float64(minInt(distX, distY))

			if dist < blendRadius {
				// Gradual transparency at the edges
				alpha := uint8((dist / blendRadius) * 255)
				mask.SetAlpha(i, j, color.Alpha{A: alpha})
			} else {
				// The core of the ROI remains fully opaque
				mask.SetAlpha(i, j, color.Alpha{A: 255})
			}
		}
	}

	// 3. Overlay the ROI using the mask for seamless blending
	dstRect := roiBounds.Add(image.Pt(x, y))
	draw.DrawMask(result, dstRect, roi, image.Point{}, mask, image.Point{}, draw.Over)

	return result
}

// Helper function for the blending math
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

package reconstruction

import (
	"image"
	"image/draw"
)

// Stitch perfectly overlays the transparent PNG silhouette onto the background
func Stitch(background image.Image, roi image.Image, x, y int) image.Image {
	if background == nil {
		return roi
	}

	bounds := background.Bounds()
	result := image.NewRGBA(bounds)

	// 1. Draw the blurred background
	draw.Draw(result, bounds, background, image.Point{}, draw.Src)

	// 2. Draw the transparent silhouette perfectly at the designated X, Y coordinates
	dstRect := image.Rect(x, y, x+roi.Bounds().Dx(), y+roi.Bounds().Dy())
	draw.Draw(result, dstRect, roi, image.Point{}, draw.Over)

	return result
}

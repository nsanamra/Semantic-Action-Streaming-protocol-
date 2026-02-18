package reconstruction

import (
	"image"
	"image/draw"

	"github.com/disintegration/imaging"
)

func StitchFrame(background image.Image, roi image.Image, x, y int) image.Image {
	// 1. Create a canvas from the background
	bounds := background.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, background, image.Point{}, draw.Src)

	// 2. Enhance ROI edges if needed (Optional sharpening)
	enhancedROI := imaging.Sharpen(roi, 0.5)

	// 3. Paste high-quality ROI over the blurry background
	// Position is defined by the coordinates sent in the SASP Header
	roiBounds := image.Rect(x, y, x+enhancedROI.Bounds().Dx(), y+enhancedROI.Bounds().Dy())
	draw.Draw(dst, roiBounds, enhancedROI, image.Point{}, draw.Over)

	return dst
}

package main

import (
	"flag"
	"image"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"

	"image/color"
	_ "image/jpeg" // needed to decode jpeg images
	"image/png"    // needed to decode png images

	pigo "github.com/esimov/pigo/core"
)

const (
	minSize     int     = 100
	maxSize     int     = 1000
	shiftFactor float64 = 0.1
	scaleFactor float64 = 1.1
	threshold   float32 = 10
)

var pFile = flag.String("pprof", "", "profiling filename")
var inDir = flag.String("in", "", "path to image directory")
var outDir = flag.String("out", "", "path to output directory")
var cascadeFileName = flag.String("casc", "", "path to cascade file")

func main() {
	flag.Parse()
	if *pFile != "" {
		f, err := os.Create(*pFile)
		if err != nil {
			log.Fatal(err)
		}
		if err = pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		defer pprof.StopCPUProfile()
	}
	if inDir == nil || outDir == nil || cascadeFileName == nil || *inDir == "" || *outDir == "" || *cascadeFileName == "" {
		log.Fatal("Missing parameters")
	}

	cascadeFile, err := ioutil.ReadFile(*cascadeFileName)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %v", err)
	}

	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := pigo.NewPigo().Unpack(cascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %s", err)
	}

	angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

	if err := os.MkdirAll(*outDir, 0755); err != nil {
		log.Fatal(err)
	}

	filepath.Walk(*inDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Fatal(err)
		}

		if !strings.HasSuffix(info.Name(), ".png") {
			return nil
		}

		num := strings.TrimLeft(strings.TrimRight(info.Name(), ".png"), "img")

		src, err := pigo.GetImage(path)
		if err != nil {
			log.Fatalf("Cannot open the image file: %v", err)
		}

		pixels := pigo.RgbToGrayscale(src)
		cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

		cParams := pigo.CascadeParams{
			MinSize:     minSize,
			MaxSize:     maxSize,
			ShiftFactor: shiftFactor,
			ScaleFactor: scaleFactor,

			ImageParams: pigo.ImageParams{
				Pixels: pixels,
				Rows:   rows,
				Cols:   cols,
				Dim:    cols,
			},
		}

		// Run the classifier over the obtained leaf nodes and return the detection results.
		// The result contains quadruplets representing the row, column, scale and detection score.
		dets := classifier.RunCascade(cParams, angle)

		// Calculate the intersection over union (IoU) of two clusters.
		dets = classifier.ClusterDetections(dets, 0.2)

		red := color.RGBA{255, 0, 0, 255}
		for _, det := range dets {
			if det.Q < threshold {
				continue
			}

			Rect(src, red, det.Col-det.Scale/2, det.Row-det.Scale/2, det.Col+det.Scale/2, det.Row+det.Scale/2)
		}

		f, err := os.Create(*outDir + "/img" + num + ".png")
		if err != nil {
			log.Fatal(err)
		}

		defer f.Close()

		if err := png.Encode(f, src); err != nil {
			log.Fatal(err)
		}

		return nil
	})
}

// src: https://stackoverflow.com/questions/28992396/draw-a-rectangle-in-golang
// HLine draws a horizontal line
func HLine(img *image.NRGBA, col color.Color, x1, y, x2 int) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine draws a veritcal line
func VLine(img *image.NRGBA, col color.Color, x, y1, y2 int) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func Rect(img *image.NRGBA, col color.Color, x1, y1, x2, y2 int) {
	HLine(img, col, x1, y1, x2)
	HLine(img, col, x1, y2, x2)
	VLine(img, col, x1, y1, y2)
	VLine(img, col, x2, y1, y2)
}

// Package TrailCamSorter provides functionality to sort and process trail camera footage.
// The TrailCamSorter struct provides methods to read, sort, and process trail camera data
// for further analysis or storage. This package also includes utility functions for handling
// common tasks such as file I/O and timestamp manipulation.
package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/otiai10/gosseract"
	log "github.com/sirupsen/logrus"
	"gocv.io/x/gocv"
)

var (
	errMissingInputDir      = errors.New("please specify input directory")
	errMissingOutputDir     = errors.New("please specify output directory")
	errLimitReached         = errors.New("limit reached")
	errFailedToReadFrame    = errors.New("failed to read frame from video")
	errJoinedImageIsEmpty   = errors.New("joined image is empty")
	errOCRReturnedEmptyText = errors.New("OCR returned empty text")
	errInvalidTrailCamData  = errors.New("invalid TrailCamData")
	errImageWrite           = errors.New("failed to write image to file")
)

// TrailCamSorter is a struct that represents a trail camera video file sorter.
type TrailCamSorter struct {
	Params SorterParams // holds the command line flags
}

// SorterParams contains parameters for the TrailCamSorter.
type SorterParams struct {
	InputDir  string // the input directory containing video files
	OutputDir string // the output directory for sorted video files
	DryRun    bool   // if true, the files will not be moved
	Debug     bool   // enables debug mode
	Limit     int    // limits the number of files processed
	Workers   int    // the number of workers used to process files
}

// TrailCamData a struct that contains the extracted data from a Trail Cam image.
type TrailCamData struct {
	Timestamp  time.Time // The timestamp of the observation (including both time and date).
	CameraName string    // The name of the camera that captured the observation.
}

// BoundingBox represents a rectangular region in an image, identified by a label string and a corresponding image.Rectangle.
type BoundingBox struct {
	Label string          // Label associated with this bounding box.
	Rect  image.Rectangle // Rectangle specifying the region in the image.
}

// Entry point of the TrailCamSorter program.
// Creates a new instance.
// Parses the command line arguments.
// Processes the files.
// Removes empty directories.
// Prints a message when done.
func main() {
	// Start the timer
	start := time.Now()

	log.SetOutput(os.Stdout)
	log.SetLevel(log.InfoLevel)

	// Create a new TrailCamSorter instance
	tcs := &TrailCamSorter{}

	// Parse the command line arguments
	err := tcs.parseFlags()
	if err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Fatal("Error parsing flags")
	}

	// Change logging level if debugging.
	if tcs.Params.Debug {
		log.SetLevel(log.DebugLevel)
	}

	// Process the files
	err = tcs.processFiles()
	if err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Fatal("Error processing files")
	}

	// Remove all empty directories in InputDir
	if err := tcs.removeEmptyDirs(); err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Error("Error removing empty directories")
	}

	log.WithFields(log.Fields{
		"time_taken": time.Since(start),
	}).Info("Done.")
}

// Parses the command line flags and stores the results in the TrailCamSorter instance.
// Returns an error if required flags are not set.
func (tcs *TrailCamSorter) parseFlags() error {
	// Set command line flags
	flag.StringVar(&tcs.Params.InputDir, "input", "", "the input directory containing video files")
	flag.StringVar(&tcs.Params.OutputDir, "output", "", "the output directory for sorted video files")
	flag.BoolVar(&tcs.Params.DryRun, "dry-run", true, "if true, the files will not be moved")
	flag.BoolVar(&tcs.Params.Debug, "debug", false, "if true, enables debug mode")
	flag.IntVar(&tcs.Params.Limit, "limit", math.MaxInt32, "limits the number of files processed")
	flag.IntVar(&tcs.Params.Workers, "workers", 0, "the number of workers used to process files")

	// Parse the command line flags
	flag.Parse()

	// Check that the required input directory flag is set.
	if tcs.Params.InputDir == "" {
		return fmt.Errorf("%w", errMissingInputDir)
	}
	// Check that the required output directory flag is set.
	if tcs.Params.OutputDir == "" {
		return fmt.Errorf("%w", errMissingOutputDir)
	}

	return nil
}

// Walks through the input directory and processes all video files by calling processFile on each file.
// It returns an error if there is an error walking the input directory.
func (tcs *TrailCamSorter) processFiles() error {
	// Create a buffered channel to receive file paths
	filesChan := make(chan string, 100)

	// Use a WaitGroup to wait for all goroutines to complete
	var wg sync.WaitGroup

	// Start the workers
	for i := 0; i < tcs.Params.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range filesChan {
				if err := tcs.processFrame(path); err != nil {
					log.WithFields(log.Fields{
						"path":  path,
						"error": err,
					}).Error("Error processing file")
				}
			}
		}()
	}

	// List of directory and file names to ignore.
	ignoreNames := []string{"$RECYCLE.BIN", ".Spotlight-V100", "System Volume Information", ".fseventsd", ".Trashes", ".DS_Store"}
	// Convert the list of ignore names to a map for efficient lookup.
	ignoreMap := make(map[string]bool)
	for _, name := range ignoreNames {
		ignoreMap[name] = true
	}

	// Walk through the input directory and send each file path to the channel
	var count int
	err := filepath.Walk(tcs.Params.InputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// Check if the error is due to a permission issue and skip the directory.
			if os.IsPermission(err) {
				log.WithFields(log.Fields{
					"path":  path,
					"error": err,
				}).Warn("Skipping directory due to permission issue")
				return filepath.SkipDir
			} else {
				// Handle other errors.
				log.WithFields(log.Fields{
					"path":  path,
					"error": err,
				}).Error("Error accessing path")
				return nil
			}
		}

		if info.IsDir() {
			// Check if the directory should be ignored.
			dir := filepath.Base(path)
			if ignoreMap[dir] {
				log.WithFields(log.Fields{
					"type": "directory",
					"path": path,
				}).Warn("Skipping ignored directory")
				return filepath.SkipDir // Skip the directory
			}
			return nil
		}

		// Check if the file should be ignored.
		filename := filepath.Base(path)
		if ignoreMap[filename] {
			log.WithFields(log.Fields{
				"type": "file",
				"path": path,
			}).Warn("Skipping ignored file")
			return nil // Skip the file
		}

		if !tcs.hasVideoFileExtension(path) {
			return nil
		}

		if tcs.Params.Limit > 0 && count >= tcs.Params.Limit {
			return fmt.Errorf("%w", errLimitReached)
		}

		filesChan <- path
		count++

		return nil
	})

	if err != nil && !errors.Is(err, errLimitReached) {
		log.WithError(err).Info("Error occurred")
	}

	// Close the files channel to signal the workers to exit
	close(filesChan)

	// Wait for all workers to complete
	wg.Wait()

	log.WithFields(log.Fields{
		"count": count,
	}).Info("Processed files")

	return nil
}

// Creates an OpenCV Mat containing a blank image of the specified width and height,
// and adds the specified label to the image.
// The function returns the resulting image as an OpenCV Mat.
func (tcs *TrailCamSorter) createLabelImage(label string, width int, height int) gocv.Mat {
	blankImage := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(0, 0, 0, 0), height, width, gocv.MatTypeCV8UC3)

	// Add the box label to the blank image.
	labelColor := color.RGBA{255, 255, 255, 0}
	labelFontScale := 1.0
	labelThickness := 1
	labelStr := fmt.Sprintf("%s: ", label)
	labelOrigin := image.Point{10, 30}
	gocv.PutText(&blankImage, labelStr, labelOrigin, gocv.FontHersheySimplex, labelFontScale, labelColor, labelThickness)

	return blankImage
}

// Reads a frame from the video file, extracts the camera name,
// time and date from the image, constructs an output path for the video file
// based on this information, and moves the video file to the output path.
// Returns an error if any of these steps fail.
func (tcs *TrailCamSorter) processFrame(inputFile string) error {
	var err error
	var data TrailCamData

	// Attempt to read the frame.
	frameNumber := 0
	for frameNumber <= 10 {
		// Read a frame from the video file.
		frame, err := tcs.readFrame(inputFile, frameNumber)
		if frame == nil {
			log.WithFields(log.Fields{
				"error":        err,
				"frame_number": frameNumber,
			}).Error("Error frame is nil")
			frameNumber++
			continue
		}

		// Write frame to file for debugging.
		err = tcs.debugImages(frame, filepath.Join(filepath.Dir(inputFile), "debug", fmt.Sprintf("%s-%d-%s.png", filepath.Base(inputFile), frameNumber, "frame")))
		if err != nil {
			frameNumber++
			continue
		}

		if err != nil {
			log.WithFields(log.Fields{
				"error":        err,
				"frame_number": frameNumber,
			}).Error("Error reading frame")
			frameNumber++
			continue
		}
		defer frame.Close()

		// Create bounding boxes scaled to the width and height of the frame.
		boundingBoxes := tcs.getBoundingBoxes(frame)

		// Create the joined image.
		joined, err := tcs.createJoinedImage(inputFile, frameNumber, boundingBoxes, frame)
		if err != nil {
			log.WithFields(log.Fields{
				"error": err,
			}).Error("Error creating joined image")
			frameNumber++
			continue
		}
		defer joined.Close()

		// Perform OCR on the joined image.
		text, err := tcs.performOCR(joined)
		if err != nil {
			log.WithFields(log.Fields{
				"error":    err,
				"ocr_text": text,
			}).Error("Error performing OCR")
			frameNumber++
			continue
		}

		// Update the TrailCamData object with the extracted data.
		data = tcs.parseOCRText(data, text)

		// Validate the TrailCamData.
		err = tcs.validateTrailCamData(data)
		if err != nil {
			log.WithFields(log.Fields{
				"error":        err,
				"frame_number": frameNumber,
			}).Error("Error validating TrailCamData")

			frameNumber++
			continue
		}

		log.WithFields(log.Fields{
			"trail_cam_data": fmt.Sprintf("%+v", data),
			"frame_number":   frameNumber,
		}).Debug("OCR: Extracted data")

		// Close the joined file explicitly.
		joined.Close()

		// If OCR succeeds, break out of the retry loop.
		break
	}

	// Construct an output path for the file.
	outputPath, err := tcs.constructOutputPath(data)
	if err != nil {
		return err
	}

	// Rename the file.
	if err := tcs.renameFile(inputFile, outputPath); err != nil {
		return err
	}

	return nil
}

// Checks if the file extension is one of the supported video file extensions.
// path is the file path to be checked.
// returns true if the file has a supported video file extension, false otherwise.
func (tcs *TrailCamSorter) hasVideoFileExtension(path string) bool {
	// Ignored filenames like: ._01.avi
	matchedInvalidAvi := regexp.MustCompile(`^\._.+\.avi$`).MatchString(filepath.Base(path))
	if matchedInvalidAvi {
		return false
	}

	// Define the list of supported video file extensions
	supportedExtensions := []string{".avi", ".mp4", ".mov", ".wmv", ".mkv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".m2v", ".ts", ".mts", ".m2ts", ".vob", ".3gp"}

	// Extract the file extension from the path
	ext := strings.ToLower(filepath.Ext(path))

	// Check if the file extension is in the list of supported video file extensions
	for _, supportedExt := range supportedExtensions {
		if ext == supportedExt {
			return true
		}
	}

	return false
}

// Constructs an output path for the video file based on the extracted info.
// The output path has the format: OutputDir/CameraName/Date/CameraName-Date-Time.avi
// If a file already exists at the output path, a suffix is added to the file name.
// Returns the constructed output path and an error, if any.
func (tcs *TrailCamSorter) constructOutputPath(data TrailCamData) (string, error) {
	// Define the format for the output path
	outputPathFormat := filepath.Join(tcs.Params.OutputDir, data.CameraName, data.Timestamp.Format("2006-01-02"), "%s-%s-%s.avi")

	// Construct the output path using the camera name, date, time
	outputPath := fmt.Sprintf(outputPathFormat, data.CameraName, data.Timestamp.Format("2006-01-02"), data.Timestamp.Format("15-04-05PM"))

	// Check if a file already exists at the output path
	i := 1
	for {
		_, err := os.Stat(outputPath)
		if os.IsNotExist(err) {
			break
		}

		// If a file already exists, add a suffix and try again
		outputPath = fmt.Sprintf(outputPathFormat, data.CameraName, data.Timestamp.Format("2006-01-02"), data.Timestamp.Format("15-04-05PM")+fmt.Sprintf("-%d", i))
		i++
	}

	// Return the constructed output path
	return outputPath, nil
}

// This function renames a file from the source path to the destination path.
// If DryRun is true, it logs the operation without actually renaming the file.
// It creates the destination directory if it doesn't exist, and returns an error if any of the operations fail.
func (tcs *TrailCamSorter) renameFile(src string, dest string) error {
	// Return early if DryRun is true
	if tcs.Params.DryRun {
		log.WithFields(log.Fields{
			"type": "DRY RUN",
			"src":  src,
			"dest": dest,
		}).Info("Skip renaming file")
		return nil
	}

	// Create the destination directory
	err := os.MkdirAll(filepath.Dir(dest), os.ModePerm)
	if err != nil {
		return err
	}

	log.WithFields(log.Fields{
		"type": "RENAME",
		"src":  src,
		"dest": dest,
	}).Info("Renaming file")

	// Rename the file
	err = os.Rename(src, dest)
	if err != nil {
		return err
	}

	return nil
}

// Reads a frame from a video file at the specified frame number and returns the frame data as a Mat object.
func (tcs *TrailCamSorter) readFrame(inputFile string, frameNumber int) (*gocv.Mat, error) {
	// Open the video file
	cap, err := gocv.VideoCaptureFile(inputFile)
	if err != nil {
		return nil, err
	}
	defer cap.Close()

	// Get the total number of frames in the video
	numFrames := cap.Get(gocv.VideoCaptureFrameCount)

	// If frameNumber is greater than 0, set it as a percentage of numFrames
	if frameNumber > 0 {
		frameNumber = int(float64(frameNumber) * numFrames / 10.0)
	}

	// Ensure that frameNumber does not exceed the total number of frames in the video
	frameNumber = int(math.Min(float64(frameNumber), numFrames))

	// Move the video to the desired frame number
	cap.Set(gocv.VideoCapturePosFrames, float64(frameNumber))

	// Read the frame from the video
	frame := gocv.NewMat()
	if ok := cap.Read(&frame); !ok {
		return nil, fmt.Errorf("%w", errFailedToReadFrame)
	}

	return &frame, nil
}

// Calculates the pixel coordinates for each label defined in labelCoordinates
// and returns a slice of bounding boxes. The bounding boxes are defined by their top-left
// and bottom-right coordinates, relative to the input image dimensions provided as arguments.
func (tcs *TrailCamSorter) getBoundingBoxes(imgMat *gocv.Mat) []BoundingBox {
	// Define the label coordinates.
	labelCoordinates := map[string][]float64{
		"Timestamp":  {0.487240, 0.972685, 0.233854, 0.054630},
		"CameraName": {0.864844, 0.972685, 0.270313, 0.054630},
	}

	// Create a slice to store the bounding boxes.
	boundingBoxes := []BoundingBox{}

	// Loop through the label coordinates and create bounding boxes for each label.
	for label, coords := range labelCoordinates {
		// Calculate the pixel values for the bounding box.
		x := coords[0]
		y := coords[1]
		w := coords[2]
		h := coords[3]
		width := float64(imgMat.Cols())
		height := float64(imgMat.Rows())
		left := (x - w/2) * width
		top := (y - h/2) * height
		right := (x + w/2) * width
		bottom := (y + h/2) * height

		// Create the bounding box struct.
		boundingBox := BoundingBox{
			Label: label,
			Rect:  image.Rect(int(left), int(top), int(right), int(bottom)),
		}

		// Add the bounding box to the slice.
		boundingBoxes = append(boundingBoxes, boundingBox)
	}

	return boundingBoxes
}

// This function takes a video frame, a slice of bounding boxes, and a filepath, and returns a
// concatenated image of the cropped bounding boxes and their labels. It first creates a container image
// to hold the label and the cropped bounding box, overlays the cropped bounding box onto the label image,
// concatenates the label image and the cropped bounding boxes, and writes the joined image to the specified filepath.
// It returns an error if any of the image operations fail.
func (tcs *TrailCamSorter) createJoinedImage(inputFile string, frameNumber int, boundingBoxes []BoundingBox, frame *gocv.Mat) (*gocv.Mat, error) {
	// Initialize the joined variable with an empty image of the correct size.
	joined := gocv.NewMatWithSize(0, 0, gocv.MatTypeCV8UC3)
	// Initialize the labelImage variable with an empty image of the correct size.
	labelImage := gocv.NewMatWithSize(0, 0, gocv.MatTypeCV8UC3)
	// Loop through each bounding box and crop out the image.
	var croppedImages []gocv.Mat
	for _, box := range boundingBoxes {
		// Crop out the bounding box.
		cropped := frame.Region(box.Rect)
		defer cropped.Close()

		// Create a blank container image to hold the label and the cropped bounding box.
		labelImage = tcs.createLabelImage(box.Label, 800, 60)

		// Overlay the cropped image onto the label image.
		roi := labelImage.Region(image.Rectangle{Min: image.Point{0, 0}, Max: image.Point{labelImage.Cols(), labelImage.Rows()}})
		cropOrigin := image.Point{labelImage.Cols() - cropped.Cols(), 0}
		croppedRoi := roi.Region(image.Rectangle{Min: cropOrigin, Max: cropOrigin.Add(image.Point{cropped.Cols(), cropped.Rows()})})
		cropped.CopyTo(&croppedRoi)

		// Write cropped image to file for debugging.
		err := tcs.debugImages(&cropped, filepath.Join(filepath.Dir(inputFile), "debug", fmt.Sprintf("%s-%d-%s.png", filepath.Base(inputFile), frameNumber, box.Label)))
		if err != nil {
			continue
		}

		// Write images to file for debugging.
		err = tcs.debugImages(&labelImage, filepath.Join(filepath.Dir(inputFile), "debug", fmt.Sprintf("%s-%d-%s.png", filepath.Base(inputFile), frameNumber, box.Label+"-label")))
		if err != nil {
			continue
		}

		croppedImages = append(croppedImages, labelImage)
	}

	// Concatenate the cropped images.
	for i := 0; i < len(croppedImages); i++ {
		if joined.Empty() {
			// If joined is empty, assign it to the first cropped image.
			joined = croppedImages[i]
		} else {
			// Otherwise, concatenate the cropped image to the bottom of joined.
			gocv.Vconcat(joined, croppedImages[i], &joined)
		}
	}

	// Close the labelImage file explicitly.
	labelImage.Close()

	if joined.Empty() {
		return nil, fmt.Errorf("%w", errJoinedImageIsEmpty)
	}

	// Write joined image to file for debugging.
	err := tcs.debugImages(&joined, filepath.Join(filepath.Dir(inputFile), "debug", fmt.Sprintf("%s-%d-%s.png", filepath.Base(inputFile), frameNumber, "joined")))
	if err != nil {
		return nil, err
	}

	return &joined, nil
}

// Performs OCR on a gocv.Mat object using Tesseract.
// Returns the recognized text and any errors that occurred during the OCR process.
func (tcs *TrailCamSorter) performOCR(imgMat *gocv.Mat) (string, error) {
	// Create a new Tesseract client.
	client := gosseract.NewClient()
	defer client.Close()

	// Convert the image to grayscale.
	grayMat := gocv.NewMat()
	defer grayMat.Close()
	gocv.CvtColor(*imgMat, &grayMat, gocv.ColorBGRToGray)

	// Convert the grayscale image to a PNG byte slice.
	pngBytes, err := gocv.IMEncode(".png", grayMat)
	if err != nil {
		return "", err
	}

	// Set the image from the PNG byte slice.
	err = client.SetImageFromBytes(pngBytes.GetBytes())
	if err != nil {
		return "", err
	}

	// Set the PageSegMode to AUTO.
	err = client.SetPageSegMode(gosseract.PSM_AUTO)
	if err != nil {
		return "", err
	}

	// Set the language to English.
	err = client.SetLanguage("eng")
	if err != nil {
		return "", err
	}

	// Perform OCR on the image and return the resulting text.
	text, err := client.Text()
	if err != nil {
		return "", err
	}
	if len(text) == 0 {
		return "", fmt.Errorf("%w", errOCRReturnedEmptyText)
	}

	return text, nil
}

// Parse OCR text and update TrailCamData object.
func (tcs *TrailCamSorter) parseOCRText(data TrailCamData, ocrText string) TrailCamData {
	lines := strings.Split(ocrText, "\n")
	for _, line := range lines {
		parts := strings.Split(line, ": ")
		if len(parts) == 2 {
			label := strings.TrimSpace(parts[0])
			text := strings.TrimSpace(parts[1])
			data = tcs.updateTrailCamData(data, label, text)
		}
	}
	return data
}

// Takes in a TrailCamData object and checks if its Timestamp field is not equal to zero,
// and its CameraName field is not an empty string.
// If either of these fields fails the validation, an error is returned.
func (tcs *TrailCamSorter) validateTrailCamData(data TrailCamData) error {
	var errs []string

	// Check for valid Timestamp.
	if data.Timestamp.IsZero() {
		errs = append(errs, "field Timestamp is invalid")
	}

	// Check for empty CameraName.
	if data.CameraName == "" {
		errs = append(errs, "field CameraName is empty")
	}

	if len(errs) > 0 {
		return fmt.Errorf("%w: %s", errInvalidTrailCamData, strings.Join(errs, "; "))
	}

	return nil
}

// Updates a TrailCamData object with OCR results for a label.
func (tcs *TrailCamSorter) updateTrailCamData(data TrailCamData, label string, text string) TrailCamData {
	switch label {
	case "Timestamp":
		timestamp, err := time.Parse("03:04PM 01/02/2006", text)
		if err != nil {
			log.WithFields(log.Fields{
				"error":    err,
				"ocr_text": text,
			}).Warn("Failed to parse timestamp")
			timestamp = time.Time{}
		}
		data.Timestamp = timestamp
	case "CameraName":
		// Replace all non-alphanumeric characters with spaces.
		regex := regexp.MustCompile("[^a-zA-Z0-9]+")
		text = regex.ReplaceAllString(text, " ")
		// Trim any leading or trailing spaces.
		text = strings.TrimSpace(text)
		// Convert to lowercase.
		text = strings.ToLower(text)
		// Replace spaces with hyphens.
		text = strings.ReplaceAll(text, " ", "-")
		data.CameraName = text
	}
	return data
}

// Removes all empty directories that are subdirectories of the input directory.
func (tcs *TrailCamSorter) removeEmptyDirs() error {
	// Return early if DryRun is true
	if tcs.Params.DryRun {
		log.WithField("type", "DRY RUN").Info("Skip removing empty directories")
		return nil
	}

	// Walk through the input directory and process each file
	err := filepath.Walk(tcs.Params.InputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.WithFields(log.Fields{
				"path":  path,
				"error": err,
			}).Error("Error accessing path")
			return nil
		}

		if !info.IsDir() {
			return nil
		}

		// Check if the directory is a subdirectory of the InputDir.
		if !strings.HasPrefix(path, tcs.Params.InputDir) {
			return filepath.SkipDir
		}

		// Check if the directory has any files or subdirectories.
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}
		if len(entries) == 0 {
			if err := os.Remove(path); err != nil {
				return err
			}
			log.WithField("path", path).Info("Removed empty directory")
		}

		return nil
	})
	if err != nil {
		log.WithField("error", err).Error("Error removing empty directories")
	}

	return nil
}

// Write image to file for debugging.
func (tcs *TrailCamSorter) debugImages(imgMat *gocv.Mat, filename string) error {
	if !tcs.Params.Debug {
		return nil
	}

	// Get the directory path from the filename.
	dir := filepath.Dir(filename)

	// Create the directory if it doesn't exist.
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err := os.MkdirAll(dir, 0o755)
		if err != nil {
			return err
		}
	}

	// Save the image to a file.
	success := gocv.IMWrite(filename, *imgMat)
	if !success {
		return fmt.Errorf("%w", errImageWrite)
	}

	log.WithField("filename", filename).Debug("Debug images written")

	return nil
}

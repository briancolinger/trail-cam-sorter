package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/otiai10/gosseract"
	log "github.com/sirupsen/logrus"
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

// A struct that contains the extracted data from a Trail Cam image.
type TrailCamData struct {
	Timestamp       time.Time // The timestamp of the observation (including both time and date).
	Temperature     int       // The temperature in degrees Celsius or Fahrenheit.
	TemperatureUnit string    // The temperature unit (either "C" or "F").
	CameraName      string    // The name of the camera that captured the observation.
}

// Declare constants for the frame dimensions and crop margins.
const (
	FrameWidth1920  = 1920 // Width of frame in pixels for 1920x1080 video.
	FrameHeight1080 = 1080 // Height of frame in pixels for 1920x1080 video.
	FrameWidth1280  = 1280 // Width of frame in pixels for 1280x720 video.
	FrameHeight720  = 720  // Height of frame in pixels for 1280x720 video.
	CropMargin60    = 60   // Number of pixels to crop from top and bottom for 1920x1080 video.
	CropMargin1200  = 1200 // Number of pixels to crop from left and right for 1920x1080 video.
	CropMargin40    = 40   // Number of pixels to crop from top and bottom for 1280x720 video.
	CropMargin800   = 800  // Number of pixels to crop from left and right for 1280x720 video.
)

// ErrUnsupportedImageDimensions is an error that is returned when the image dimensions are unsupported.
var ErrUnsupportedImageDimensions = errors.New("unsupported image dimensions")

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
	log.SetLevel(log.DebugLevel)

	// Create a new TrailCamSorter instance
	tcs := &TrailCamSorter{}

	// Parse the command line arguments
	err := tcs.parseFlags()
	if err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Fatal("Error parsing flags")
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

	// Check that the required inputDir and outputDir flags are set
	if tcs.Params.InputDir == "" || tcs.Params.OutputDir == "" {
		return fmt.Errorf("please specify inputdir and outputdir")
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
				if err := tcs.processFile(path); err != nil {
					log.WithFields(log.Fields{
						"path":  path,
						"error": err,
					}).Error("Error processing file")
				}
			}
		}()
	}

	// Walk through the input directory and send each file path to the channel
	var count int
	err := filepath.Walk(tcs.Params.InputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.WithFields(log.Fields{
				"path":  path,
				"error": err,
			}).Error("Error accessing path")
			return nil
		}

		if info.IsDir() {
			return nil
		}

		if !tcs.hasVideoFileExtension(path) {
			return nil
		}

		if tcs.Params.Limit > 0 && count >= tcs.Params.Limit {
			return fmt.Errorf("limit reached")
		}

		filesChan <- path
		count++

		return nil
	})

	if err != nil && err.Error() != "limit reached" {
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

// Reads a frame from the video file, extracts the camera name,
// time and date from the image, constructs an output path for the video file
// based on this information, and moves the video file to the output path.
// Returns an error if any of these steps fail.
func (tcs *TrailCamSorter) processFile(inputFile string) error {
	var err error

	defer func() {
		if err != nil {
			log.WithFields(log.Fields{
				"input_file": inputFile,
				"error":      err,
			}).Error("Error processing file")
		} else {
			log.WithFields(log.Fields{
				"event":      "processed_file",
				"input_file": inputFile,
			}).Info("Done processing file")
		}
	}()

	var rawText string
	for numTries := 0; numTries <= 5; numTries++ {
		log.WithFields(log.Fields{
			"event":      "processing_file",
			"input_file": inputFile,
			"num_tries":  numTries,
		}).Info("Processing file")

		// Read a frame from the video file.
		frameBytes, err := tcs.readFrame(inputFile, numTries)
		if err != nil {
			continue
		}

		// Crop the image based on its dimensions.
		var croppedImg image.Image
		width, height, err := tcs.getImageDimensions(bytes.NewReader(frameBytes))
		if err != nil {
			continue
		}

		switch {
		case width == FrameWidth1920 && height == FrameHeight1080:
			croppedImg, err = tcs.cropImage(frameBytes, CropMargin60, CropMargin1200)
		case width == FrameWidth1280 && height == FrameHeight720:
			croppedImg, err = tcs.cropImage(frameBytes, CropMargin40, CropMargin800)
		default:
			err = ErrUnsupportedImageDimensions
		}
		if err != nil {
			return err
		}

		if tcs.Params.Debug {
			tcs.debugImages(inputFile, frameBytes, croppedImg, numTries)
		}

		// Convert the image to PNG format.
		var buf bytes.Buffer
		if err := png.Encode(&buf, croppedImg); err != nil {
			return err
		}
		pngBytes := buf.Bytes()

		// Perform optical character recognition (OCR) on the PNG data to extract the text.
		rawText, err = tcs.performOCR(pngBytes)
		if err != nil {
			log.WithField("error", err).Error("Error performing OCR")
			continue
		}

		log.WithFields(log.Fields{
			"event":    "performed_ocr",
			"raw_text": rawText,
		}).Info("OCR: Extracted raw text")

		// If OCR succeeds, break out of the retry loop.
		break
	}

	// Parse data from the text.
	data, err := tcs.parseTrailCamData(rawText)
	if err != nil {
		return err
	}

	log.WithFields(log.Fields{
		"event":          "extracted_data",
		"trail_cam_data": fmt.Sprintf("%+v", data),
	}).Info("OCR: Extracted data")

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

// Reads the dimensions of an image file without loading the entire image into memory.
func (tcs *TrailCamSorter) getImageDimensions(r io.Reader) (int, int, error) {
	// Decode the image header to get the dimensions.
	config, _, err := image.DecodeConfig(r)
	if err != nil {
		return 0, 0, err
	}

	return config.Width, config.Height, nil
}

// Checks if the file extension is one of the supported video file extensions.
// path is the file path to be checked.
// returns true if the file has a supported video file extension, false otherwise.
func (tcs *TrailCamSorter) hasVideoFileExtension(path string) bool {
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

// Reads a frame from a video file at the specified time offset and returns the frame data as a PNG image.
func (tcs *TrailCamSorter) readFrame(inputFile string, retry int) ([]byte, error) {
	// Calculate the timestamp as a string in the format "00:00:00.000" based on the retry counter.
	timestamp := fmt.Sprintf("%02d:%02d:%02d.000", retry/3600, (retry/60)%60, retry%60)

	// Build the FFmpeg command to extract the frame at the specified timestamp as a grayscale PNG image and write it to a pipe.
	cmd := exec.Command("ffmpeg", "-loglevel", "error", "-ss", timestamp, "-i", inputFile, "-vframes", "1", "-f", "image2pipe", "-pix_fmt", "gray", "-c:v", "png", "-")

	// Create a buffer to hold the output from the command.
	var buf bytes.Buffer

	// Set the command's output to the buffer.
	cmd.Stdout = &buf

	// Run the FFmpeg command.
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	// Return the PNG data as a byte slice.
	return buf.Bytes(), nil
}

// Takes an image in PNG format, along with the bottom and right arguments, and returns a new image that has been cropped to the specified dimensions.
func (tcs *TrailCamSorter) cropImage(pngBytes []byte, bottom int, right int) (image.Image, error) {
	// Decode the PNG data into an image
	imgReader := bytes.NewReader(pngBytes)
	imgDecoded, err := png.Decode(imgReader)
	if err != nil {
		return nil, err
	}

	// Create a new box using the bottom and right arguments to crop the image
	box := imgDecoded.Bounds()
	box.Min.X = box.Max.X - right
	box.Min.Y = box.Max.Y - bottom

	// Crop the image using the box and return the result
	return imgDecoded.(interface {
		SubImage(r image.Rectangle) image.Image
	}).SubImage(box), nil
}

func (tcs *TrailCamSorter) performOCR(pngBytes []byte) (string, error) {
	// Create a new Tesseract client and set the image from the provided bytes
	client := gosseract.NewClient()
	defer client.Close()
	client.SetImageFromBytes(pngBytes)

	// Set the PageSegMode to AUTO
	if err := client.SetPageSegMode(gosseract.PSM_AUTO); err != nil {
		return "", fmt.Errorf("failed to set PageSegMode: %v", err)
	}

	//  Set the language to English
	if err := client.SetLanguage("eng"); err != nil {
		return "", fmt.Errorf("failed to set Language: %v", err)
	}

	// Perform OCR on the image and return the resulting text
	text, err := client.Text()
	if err != nil {
		return "", err
	}
	if len(text) == 0 {
		return "", fmt.Errorf("OCR failed: no text returned")
	}

	return text, nil
}

// Parses the input text and extracts relevant data into a TrailCamData struct.
// Returns the TrailCamData struct and an error, if any.
func (tcs *TrailCamSorter) parseTrailCamData(text string) (TrailCamData, error) {
	// Compile the regular expression to match the required patterns.
	pattern := `(?P<timestamp>\d{2}:\d{2}[AP]M\s+\d{1,2}/\d{1,2}/\d{4})\s+(?P<temp>(?:-)?(?:[A-Za-z]\d+|\d+))\s*(?P<temp_unit>°[CF])\s*[^\w\r\n]*(?P<camera_name>\w+(?:\s+\w+)*)$`
	regex := regexp.MustCompile(pattern)

	// Extract the named capture groups from the input text.
	match := regex.FindStringSubmatch(text)
	if match == nil {
		return TrailCamData{}, fmt.Errorf("failed to extract data from text: %s", text)
	}
	groups := make(map[string]string)
	for i, name := range regex.SubexpNames() {
		if i != 0 && name != "" {
			groups[name] = match[i]
		}
	}

	// Parse the timestamp and temperature
	ts, err := time.Parse("03:04PM 01/02/2006", groups["timestamp"])
	if err != nil {
		return TrailCamData{}, fmt.Errorf("failed to parse timestamp: %v", err)
	}

	var temp int = 0
	tempString := groups["temp"]
	if tempString != "" {
		val, err := strconv.Atoi(tempString)
		if err != nil {
			// Set temp to 0 if temperature value is not valid.
			temp = 0
		} else {
			temp = val
		}
	}

	var unit string
	unitString := groups["temp_unit"]
	if unitString != "" {
		// Remove the degree symbol from the temperature unit.
		unitString = strings.TrimPrefix(unitString, "°")
		unit = unitString
	} else {
		// Set the temperature unit to "F" by default.
		unit = "F"
	}

	// Extract the camera name
	cn := strings.TrimSpace(groups["camera_name"])
	// Sometimes the parsed camera name is prefixed with a single letter and a space.
	// This happens because the OCR is seeing a symbol that represents the moon phase.
	// Sometimes the moon phase symbol looks like a capital O.
	// This is kinda hacky, but just trim off a single letter and a space.
	if len(cn) > 1 && unicode.IsLetter(rune(cn[0])) && cn[1] == ' ' {
		cn = cn[2:]
	}
	cn = strings.ReplaceAll(cn, " ", "-")
	cn = strings.ToLower(cn)

	// Create a new TrailCamData struct and return it.
	data := TrailCamData{
		Timestamp:       ts,
		Temperature:     temp,
		TemperatureUnit: unit,
		CameraName:      cn,
	}
	return data, nil
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

// Write images to file for debugging purposes.
func (tcs *TrailCamSorter) debugImages(inputFile string, frame []byte, croppedImg image.Image, numTries int) {
	// Create the debug directory
	debugDir := filepath.Join(tcs.Params.InputDir, "debug")
	err := os.MkdirAll(debugDir, os.ModePerm)
	if err != nil {
		log.WithFields(log.Fields{
			"debugDir": debugDir,
			"error":    err,
		}).Debug("Error creating debug directory")
	}

	// Get the input file name without the extension
	inputFileName := strings.TrimSuffix(filepath.Base(inputFile), filepath.Ext(inputFile))

	// Decode the PNG image data into an image.Image object
	frameImg, err := png.Decode(bytes.NewReader(frame))
	if err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Debug("Error decoding frame")
		return
	}

	// Create the frame file
	frameFilePath := filepath.Join(debugDir, fmt.Sprintf("%s-frame-%d.png", inputFileName, numTries))
	frameFile, err := os.Create(frameFilePath)
	if err != nil {
		log.WithFields(log.Fields{
			"frame": frameFilePath,
			"error": err,
		}).Debug("Error creating file for frame")
	}
	defer frameFile.Close()

	// Write the frame to a file
	if err := png.Encode(frameFile, frameImg); err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Debug("Error writing frame to file")
	}

	// Create the cropped image file
	croppedFilePath := filepath.Join(debugDir, fmt.Sprintf("%s-cropped-%d.png", inputFileName, numTries))
	croppedFile, err := os.Create(croppedFilePath)
	if err != nil {
		log.WithFields(log.Fields{
			"cropped": croppedFilePath,
			"error":   err,
		}).Debug("Error creating file for cropped image")
	}
	defer croppedFile.Close()

	// Write the cropped image to a file
	if err := png.Encode(croppedFile, croppedImg); err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Debug("Error writing cropped image to file")
	}

	log.WithFields(log.Fields{
		"frame":   frameFilePath,
		"cropped": croppedFilePath,
	}).Debug("Debug images written")
}

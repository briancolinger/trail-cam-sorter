# `trail-cam-sorter`

With `trail-cam-sorter` you can efficiently process videos files concurrently, extract essential metadata such as date, time, and camera name, and organize the files based on the extracted metadata.

## Key Features

- Concurrent video processing: Process a large number of videos at once to save time.
- Metadata extraction: Extract essential metadata such as date, time, and camera name.
- Automated organization: Organize videos into directories based on the extracted metadata for easy file management.
- Wide video format support: Support for a broad range of video formats, including AVI, MP4, and more, to ensure compatibility with various camera models.

## How it Works

`trail-cam-sorter` utilizes [`GoCV`](https://gocv.io/) to read a frame from each video file, detects bounding boxes that contain the timestamp and camera name using object detection techniques, and then uses OCR (Optical Character Recognition) to extract text from the bounding boxes. Based on the extracted metadata, the video files are renamed and organized into a directory structure with the following format: CameraName/Date/CameraName-Date-Time.avi.

## Getting Started

1. Clone the repository to your local machine.
2. Navigate to the `trail-cam-sorter` directory in your terminal.

### Install:

```bash
make install
```

### Run:

```bash
trail-cam-sorter --input=/path/to/input --output=/path/to/output
```

### Optional Parameters

- --dry-run=true to skip renaming the files.
- --limit=10 to process only the first 10 files.
- --debug=true to write debug images to file.
- --workers=2 to process files concurrently.

### Run with optional parameters:

```bash
trail-cam-sorter --input=/path/to/input --output=/path/to/output --dry-run=true --limit=10 --debug=true --workers=2
```

### Remove:

```bash
make clean
```

## License

`trail-cam-sorter` is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html). Feel free to use, modify, and distribute this software as needed.

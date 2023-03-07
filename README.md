# `trail-cam-sorter`

`trail-cam-sorter` is a Go application that streamlines the management and sorting of large collections of trail camera video files. With `trail-cam-sorter`, you can efficiently process videos concurrently, extract essential metadata such as date, time, and camera name, and organize files automatically based on the extracted metadata.

## Key Features

- Concurrent video processing: Efficiently process a large number of trail camera videos at once to save time.
- Metadata extraction: Automatically extract essential metadata such as date, time, and camera name to quickly find and sort videos.
- Automated organization: Automatically organize videos into directories based on the extracted metadata for easy file management.
- Wide video format support: Support for a broad range of video formats, including AVI, MP4, and more, to ensure compatibility with various camera models.

## How it Works

`trail-cam-sorter` utilizes [`GoCV`](https://gocv.io/) to read a frame from each trail camera video file, detects bounding boxes that contain the timestamp and camera name using object detection techniques, and then uses OCR (Optical Character Recognition) to extract text from the bounding boxes. Based on the extracted metadata, the video files are renamed and organized into a directory structure with the following format: CameraName/Date/CameraName-Date-Time.avi.

## Getting Started

### Running the Docker container

To quickly get started with Trail Cam Sorter, follow these steps:

1. Clone the `trail-cam-sorter` repository to your local machine.
2. Navigate to the `trail-cam-sorter` directory.
3. Copy the `.env.example` file to `.env`.
4. Open the `.env` file and update the `INPUT_DIR` and `OUTPUT_DIR` variables to specify the input and output paths for your trail camera video files.
5. Execute the `docker-compose up` command to start the Trail Cam Sorter container.

### Running outside of Docker

If you prefer to run `trail-cam-sorter` outside of the Docker container, you can use the following command:

```bash
go run main.go --input=/path/to/input --output=/path/to/output --dry-run=false --limit=0 --debug=false --workers=4
```

This command processes all videos in the specified input directory, extracts metadata from each video, and sorts the files into a directory structure based on the camera name and date.

- Change --dry-run=true to skip renaming the files.
- Change --limit=10 to process only the first 10 files.
- Change --debug=true to write extracted frames and cropped images to file.
- Change --workers=1 to process files one at a time.

## License

`trail-cam-sorter` is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html). Feel free to use, modify, and distribute this software as needed.

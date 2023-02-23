# Trail Cam Sorter

Trail Cam Sorter is a Go application that helps you manage and sort large numbers of trail camera video files. With Trail Cam Sorter, you can easily process videos concurrently, extract metadata, and organize files based on the extracted metadata.

## Features

- Concurrently process large numbers of trail camera videos
- Extract metadata from each video, including time, date, and camera name
- Organize video files into directories based on the camera name and date
- Support for a wide range of video formats, including AVI, MP4, and more

## How it Works

Trail Cam Sorter uses `ffmpeg` to extract the first frame from each video, crops the image to extract the time, date, and camera name banner, and then uses OCR to extract metadata. The video files are then renamed and moved into a directory structure with the following format: `CameraName/Date/CameraName-Date-Time.avi`.

## Running the Docker container

To get started with Trail Cam Sorter, follow these steps:

1. Clone the Trail Cam Sorter repository to your local machine
2. `cd` to the `trail-cam-sorter` directory
3. Copy `.env.example` to `.env`
4. Edit `.env` and update the input and output paths
5. Execute the `docker-compose up` command

## Running outside of Docker

Here are some examples of how to use Trail Cam Sorter outside of the Docker container:

```bash
go run main.go --input=/trail-cams/unsorted --output=/trail-cams/sorted --dry-run=false --limit=0 --debug=false --workers=4
```

This command processes all videos in the /trail-cams directory, extracts metadata from each video, and sorts the files into a directory structure based on the camera name and date.

- Change --dry-run=true to skip renaming the files.
- Change --limit=10 to only process 10 files.
- Change --debug=true to write extracted frame and cropped image to file.
- Change --workers=1 to process files one at a time.

## License

Trail Cam Sorter is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html). Feel free to use, modify, and distribute this software as needed.

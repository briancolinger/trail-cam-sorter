FROM otiai10/gosseract:latest

# Install the ffmpeg package
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory to /go/src/app
WORKDIR /go/src/app

# Add the l alias to .bashrc
RUN echo 'alias l="ls -lah"' >> ~/.bashrc

# Set the args for the main app
ARG input=/trail-cams/unsorted/100MUDDY1
ARG output=/trail-cams/sorted
ARG dry-run=true
ARG limit=10
ARG debug=false
ARG workers=1

ENV INPUT_DIR=$input_dir
ENV OUTPUT_DIR=$output_dir
ENV DRY_RUN=$dry_run
ENV LIMIT=$limit
ENV DEBUG=$debug
ENV WORKERS=$workers

# Copy the app files into the container
COPY . .

RUN go get -d -v ./...

# Set the entry point for the container
CMD ["go", "run", "main.go"]

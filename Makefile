APP_NAME := trail-cam-sorter

.PHONY: all build install clean

all: build

build:
	@go build -o $(APP_NAME)

install: build
	@go install
	@INSTALL_DIR=$$(go env GOBIN); \
	if [ -z "$$INSTALL_DIR" ]; then \
		echo "Installation directory is not set. "; \
		echo "Add 'export GOBIN=\$$HOME/go/bin' to your PATH environment variable and try again."; \
	else \
		echo "Installed $$INSTALL_DIR/$(APP_NAME)"; \
	fi

clean:
	@go clean -i -r
	@if [ -f "$(APP_NAME)" ]; then rm "$(APP_NAME)"; fi

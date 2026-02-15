BINARY  := selfext
MODULE  := github.com/onusrat/selfext
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "none")
DATE    := $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')
LDFLAGS := -s -w -X main.Version=$(VERSION) -X main.CommitHash=$(COMMIT) -X main.BuildDate=$(DATE)

PLATFORMS := linux/amd64 linux/arm64 linux/riscv64 darwin/arm64

.PHONY: build build-all install clean vet fmt help

build: ## Build for current platform
	go build -ldflags "$(LDFLAGS)" -o $(BINARY) .

build-all: ## Cross-compile for all platforms
	@mkdir -p build
	@for platform in $(PLATFORMS); do \
		os=$${platform%/*}; arch=$${platform#*/}; \
		echo "Building $$os/$$arch..."; \
		GOOS=$$os GOARCH=$$arch go build -ldflags "$(LDFLAGS)" \
			-o build/$(BINARY)-$$os-$$arch . ; \
	done
	@echo "Done. Binaries in build/"

install: build ## Install to ~/.local/bin
	@mkdir -p ~/.local/bin
	cp $(BINARY) ~/.local/bin/$(BINARY)
	@echo "Installed to ~/.local/bin/$(BINARY)"

clean: ## Remove build artifacts
	rm -rf build/ $(BINARY)

vet: ## Run go vet
	go vet ./...

fmt: ## Run go fmt
	go fmt ./...

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

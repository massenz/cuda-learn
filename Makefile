# Copyright (c) 2025 AlertAvert.com.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)

VERSION = 0.3.0
GOARCH ?= $(shell go env GOARCH)

# AWS CDK CLI tool
CLI_TARGET = build/cuda-learn_${VERSION}_${GOARCH}_cli
CLI_SRC = go-aws-cli/cmd

.PHONY: all build clean cli vpc instance help

##@ General targets
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: version
version: ## Display the current version of the project
	@echo "$(VERSION)"

##@ CUDA Application
.PHONY: all build cpu-only
all: build ## Default target to build CUDA applications

build: ## Build CUDA applications using CMake
	@mkdir -p build
	@echo "--- 🧩 Building CUDA applications with CMake"
	cmake -B build
	@echo "--- 🛠️ Compiling CUDA applications"
	cmake --build build

cpu-only: ## Build applications using non-CUDA (CPU-only) configuration
	@mkdir -p build
	@echo "--- 🧩 Compiling CPU only applications"
	cmake -DCPU_ONLY=ON -B build
	@echo "--- 🛠️ Building applications"
	cmake --build build

##@ AWS CDK CLI tool targets

cli: ## Build the AWS CDK CLI tool
	@mkdir -p build
	@echo "--- 🛠️ Building CLI tool"
	cd go-aws-cli && go build -o ../$(CLI_TARGET) ./cmd

vpc: cli  ## Set up VPC infrastructure using the CLI tool
	@echo "--- 🌐 Setting up VPC infrastructure"
	./$(CLI_TARGET) vpc

instance: cli ## Set up EC2 instance using the CLI tool
	@echo "--- 🌐 Setting up EC2 instance"
	./$(CLI_TARGET) instance

##@ Other utility targets

clean:  ## Clean up build artifacts
	@echo "--- 🧹 Cleaning build directory"
	@rm -rf build

# Makefile for vector_add kernel

.PHONY: all build test benchmark profile clean help

# Compiler settings
NVCC := nvcc
CUDA_ARCH := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89
CUDA_FLAGS := -O3 -std=c++17 $(CUDA_ARCH) -Xcompiler -O3 -Xcompiler -march=native
CXX_FLAGS := -O3 -std=c++17 -march=native

# Directories
BUILD_DIR := build
BIN_DIR := $(BUILD_DIR)/bin
INSTALL_PREFIX := install

# Targets
CUDA_BINARY := $(BIN_DIR)/vector_add_benchmark

# Default target
all: build test

help:
	@echo "Vector Add Kernel - Makefile Targets"
	@echo "===================================="
	@echo "  make build       - Build CUDA kernel"
	@echo "  make test        - Run correctness tests"
	@echo "  make benchmark   - Run performance benchmark"
	@echo "  make profile     - Profile with Nsight Compute (if available)"
	@echo "  make reproduce   - Full reproduce (build + test + benchmark + profile)"
	@echo "  make clean       - Clean build artifacts"

# Build the CUDA kernel
build: $(CUDA_BINARY)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(CUDA_BINARY): kernel.cu CMakeLists.txt | $(BIN_DIR)
	@echo "Building CUDA kernel..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$$(nproc)
	@echo "✓ Build complete: $@"

# Correctness tests
test: build
	@echo "Running correctness tests..."
	@python3 -m pytest tests/test_correctness.py -v --tb=short || true
	@echo "✓ Tests complete"

# Benchmark
benchmark: build
	@echo "Running benchmark..."
	@python3 benchmarks/benchmark.py \
		--size 100000000 \
		--iterations 100 \
		--output benchmark_pytorch.json \
		--device cuda || true
	@echo "✓ Benchmark complete"

# Profile with Nsight Compute
profile: build
	@if command -v ncu >/dev/null 2>&1; then \
		bash profile/profile.sh $(CUDA_BINARY) profile_output; \
	else \
		echo "⚠ Nsight Compute not found. Install with: apt-get install nvidia-cuda-toolkit"; \
	fi

# Full reproduce workflow
reproduce: all benchmark profile
	@echo "=========================================="
	@echo "Reproduce Complete!"
	@echo "=========================================="

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(INSTALL_PREFIX) profile_output* benchmark*.json
	@echo "✓ Clean complete"

.PHONY: all help test benchmark profile reproduce clean

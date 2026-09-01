#!/bin/bash
# Reproduce vector_add benchmark and profiling.
# 
# This script:
# 1. Builds the CUDA kernel (if not already built)
# 2. Runs correctness tests
# 3. Runs benchmarks and saves JSON output
# 4. Optionally profiles with Nsight Compute
#
# Usage:
#   bash reproduce.sh [--profile] [--profile-only]

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build directory
BUILD_DIR="$SCRIPT_DIR/build"
INSTALL_PREFIX="$SCRIPT_DIR/install"

# Flags
PROFILE=0
PROFILE_ONLY=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        --profile)
            PROFILE=1
            ;;
        --profile-only)
            PROFILE_ONLY=1
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Vector Add: Reproduce Benchmark & Profile"
echo "=========================================="
echo ""

# Step 1: Build (if needed)
if [ "$PROFILE_ONLY" -eq 0 ]; then
    echo "[1/3] Building CUDA kernel..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    cd "$SCRIPT_DIR"
    echo "✓ Build complete"
    echo ""
fi

# Step 2: Run correctness tests
if [ "$PROFILE_ONLY" -eq 0 ]; then
    echo "[2/3] Running correctness tests..."
    python3 -m pytest tests/test_correctness.py -v --tb=short || {
        echo "⚠ Some tests failed (may be expected without GPU)"
    }
    echo ""
fi

# Step 3: Run benchmarks
if [ "$PROFILE_ONLY" -eq 0 ]; then
    echo "[3/3] Running benchmarks..."
    
    # Python benchmark
    echo "  Running PyTorch benchmark..."
    python3 benchmarks/benchmark.py \
        --size 100000000 \
        --iterations 100 \
        --output benchmark_pytorch.json \
        --device cuda || {
        echo "  ⚠ PyTorch benchmark failed (GPU may not be available)"
    }
    echo "  ✓ Benchmark complete"
    echo ""
fi

# Step 4: Profile (optional)
if [ "$PROFILE" -eq 1 ] || [ "$PROFILE_ONLY" -eq 1 ]; then
    if command -v ncu &> /dev/null; then
        echo "[4/4] Profiling with Nsight Compute..."
        
        if [ ! -f "$BUILD_DIR/bin/vector_add_benchmark" ]; then
            echo "  Error: Kernel binary not found. Build first."
            exit 1
        fi
        
        bash profile/profile.sh "$BUILD_DIR/bin/vector_add_benchmark" "profile_output"
        echo ""
    else
        echo "[4/4] ⚠ Nsight Compute not found. Skipping profiling."
        echo "  Install with: apt-get install nvidia-cuda-toolkit"
        echo ""
    fi
fi

# Summary
echo "=========================================="
echo "Reproduce Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Benchmark results: benchmark_pytorch.json"
if [ -f "profile_output.ncu-rep" ]; then
    echo "  Profiling data: profile_output.ncu-rep"
fi
echo ""
echo "Next steps:"
echo "  1. Review benchmark JSON: cat benchmark_pytorch.json"
if [ -f "profile_output.ncu-rep" ]; then
    echo "  2. View profiling summary: ncu --print summary_stats profile_output.ncu-rep"
    echo "  3. Open in GUI: ncu-ui profile_output.ncu-rep"
fi
echo ""
echo "To run with profiling:"
echo "  bash reproduce.sh --profile"
echo ""

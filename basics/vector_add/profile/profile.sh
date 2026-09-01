#!/bin/bash
# Profile vector_add kernel with Nsight Compute
#
# Usage:
#   bash profile.sh <kernel_binary> [output_name]
#
# Example:
#   bash profile.sh ./build/vector_add profile_output

set -e

KERNEL_BINARY="${1:-./build/vector_add}"
OUTPUT_NAME="${2:-profile_output}"

if [ ! -f "$KERNEL_BINARY" ]; then
    echo "Error: Kernel binary not found: $KERNEL_BINARY"
    exit 1
fi

if ! command -v ncu &> /dev/null; then
    echo "Error: Nsight Compute (ncu) not found. Install NVIDIA Nsight Compute."
    exit 1
fi

echo "Profiling $KERNEL_BINARY..."
echo "Output: ${OUTPUT_NAME}.ncu-rep"
echo ""

# Key metrics for understanding bottlenecks:
# - sm__utilization: Streaming Multiprocessor utilization
# - sm__warps_active: Active warps per SM
# - memory_l1_system_coherence_requests: Memory transactions
# - memory_l1_system_coherence_requests_miss_rate: L1 miss rate
# - smsp__inst_executed_per_inst_issued: Instruction throughput

METRICS="sm__utilization.avg,sm__warps_active.avg,memory_l1_system_coherence_requests,memory_l1_system_coherence_requests_miss_rate,smsp__inst_executed_per_inst_issued"

# Run Nsight Compute with selected metrics
# -f: Force overwrite output file
# -o: Output file name
# --metrics: Comma-separated list of metrics
# --section: Can also specify sections like "SpeedOfLight" or "MemoryWorkloadAnalysis"
ncu -f \
    -o "${OUTPUT_NAME}" \
    --metrics "${METRICS}" \
    "$KERNEL_BINARY" \
    --size 100000000 \
    --iterations 100

echo ""
echo "Profiling complete. Output: ${OUTPUT_NAME}.ncu-rep"
echo ""
echo "To view results:"
echo "  ncu --print summary_stats ${OUTPUT_NAME}.ncu-rep"
echo ""
echo "To open in GUI:"
echo "  ncu-ui ${OUTPUT_NAME}.ncu-rep"

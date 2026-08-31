# GPU kernel optimization principles

## Start with a bottleneck

A kernel is not optimized by adding threads indiscriminately. Measure whether the
operation is limited by memory bandwidth, arithmetic throughput, synchronization, or
launch latency. A roofline estimate compares arithmetic intensity (operations per
byte) with the device's compute-to-bandwidth ratio. Elementwise operations usually
have low intensity and benefit from coalescing and fusion; matrix operations usually
have enough reuse to benefit from tiling and matrix instructions.

## Map work to the execution model

CUDA launches a grid of thread blocks. Threads in a warp execute the same instruction
path, so adjacent lanes should normally access adjacent addresses. A Triton program is
a block-level description that the compiler maps to warps. The block size is a
compile-time choice: it enables unrolling and vectorized memory operations but raises
register and shared-memory demand when excessive.

## Reduce data movement

Global memory is high-latency and finite-bandwidth. Re-reading an activation or
materializing an intermediate costs bytes even when the arithmetic is cheap. Fusion
keeps a value in registers between operations, at the cost of more registers and
potentially lower occupancy. Tiling reuses a loaded tile across multiple operations.

## Preserve numerical meaning

Reductions are order-dependent in floating point. Accumulate softmax statistics and
normalization moments in FP32 when inputs are FP16/BF16, and use stable transforms
such as subtracting the row maximum before exponentiation. Tolerances must be
specified before timing; otherwise a faster but inaccurate kernel can be selected.

## Validate and measure

Use awkward sizes to exercise masks and tails. Warm up JIT compilation, synchronize
GPU work, and report a median plus a tail percentile. A timing without device,
driver, dtype, shape, and software revision is not reproducible evidence.

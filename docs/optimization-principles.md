# GPU kernel optimization principles

This note supplies the vocabulary used by the individual KernelOpt examples. A
**kernel** is a small GPU program that applies an operation to many data elements.
The goal is not to use the largest possible number of threads; it is to move the
right data the right number of times while keeping the GPU's arithmetic units busy.

## First ask what is slow

There are four common limits:

* **Memory bandwidth:** the kernel is waiting for bytes from device memory.
* **Arithmetic throughput:** the arithmetic units cannot perform operations quickly
  enough.
* **Synchronization:** workers must wait for one another, often during a reduction.
* **Launch latency:** the work is so small that starting the kernel is expensive.

Arithmetic intensity means “operations performed per byte transferred.” An
elementwise addition has very little arithmetic per byte and is usually
bandwidth-bound. A matrix multiplication reuses loaded values many times and is
more likely to benefit from tiling and specialized matrix units. This diagnosis is
the beginning of optimization, not an after-the-fact justification.

## How GPU work is divided

CUDA launches a **grid** of **thread blocks**. Threads in a block can cooperate;
threads are executed in groups called **warps** (typically 32 threads). When
neighboring warp lanes read neighboring addresses, the hardware can combine those
requests into efficient **coalesced** memory transactions. Branches whose lanes take
different paths cause **divergence**, leaving some lanes idle.

Triton expresses a block of related elements rather than individual threads. Its
compiler chooses the lower-level mapping, while constants such as block size remain
available for tuning. Larger blocks can improve vectorization and reduce scheduling
overhead, but consume more registers. Too many registers reduce **occupancy**, the
number of active warps that can reside on a multiprocessor.

## Why fewer memory operations often wins

Device memory is large but much slower than registers and on-chip shared memory.
Suppose one operation writes an intermediate tensor and the next operation reads it
back: the GPU has paid for a store and a load even though the intermediate has no
meaning outside the pair. **Fusion** performs both operations in one kernel and keeps
the intermediate in registers. Fusion saves traffic and a launch, but can increase
register pressure and instruction count.

**Tiling** is the complementary idea for operations with reuse. A tile is loaded
once into fast on-chip storage and used by multiple calculations. The best choice
depends on shape, layout, and the surrounding model; an isolated kernel can lose
its advantage if tensors must first be transposed.

## Numerical behavior is part of correctness

Floating-point addition is not perfectly associative. A parallel reduction can add
values in a different order from a reference implementation, producing a small
but measurable difference. Low-precision inputs such as FP16 and BF16 therefore
often use FP32 accumulators. Softmax additionally subtracts the row maximum before
exponentiating so ordinary inputs do not overflow.

Choose tolerances before measuring speed. Otherwise an inaccurate result can appear
to be an optimization. The tests in this repository deliberately include awkward
sizes so masked tails, borders, and odd dimensions are exercised.

## Measure an implementation, not a story

JIT compilation and the first memory accesses are not representative steady-state
work. Warm up, synchronize the device, collect repeated timings, and report a
median and a tail percentile. A useful result records the GPU, driver, dtype,
shape, software revision, and competing baseline. KernelOpt's catalogue is intended
to preserve those details.

## Further reading

* [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  explains grids, blocks, warps, memory coalescing, and synchronization.
* [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  develops occupancy, memory behavior, and measurement methodology.
* [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/) show how
  block-level Python descriptions become GPU kernels.

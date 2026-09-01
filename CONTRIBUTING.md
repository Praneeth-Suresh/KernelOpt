# Contributing to KernelOpt

Thank you for contributing to KernelOpt! This guide ensures that all submissions meet our standards for correctness, performance, reproducibility, and pedagogy.

---

## Core Principles

1. **Correctness First**: Every kernel must be numerically correct before it's considered fast. Timings are meaningless if the result is wrong.
2. **Reproducibility**: All benchmarks, profiling, and autotuning results must be reproducible. Document environment, hardware, seeds, and iteration counts.
3. **Pedagogy**: Code and documentation must teach. Comments explain the mapping from code to hardware; README sections guide beginners.
4. **References**: Cite the canonical works that inspired your optimizations (roofline model, memory coalescing patterns, warp primitives, etc.).

---

## Submission Types & Checklists

### 1. Adding a Foundational Kernel

**Scope**: Vector operations (vector add, dot product), dense linear algebra (matrix-vector, GEMM), convolution, reduction, prefix-sum.

**Directory Structure**:
```
basics/<kernel_name>/
  kernel.cu                 # CUDA implementation (required)
  kernel_ref.py             # PyTorch reference implementation
  kernel_triton.py          # Triton implementation (optional but recommended)
  README.md                 # Pedagogical documentation
  tests/
    test_correctness.py     # Numerical correctness tests
    test_edge_cases.py      # Edge case tests (odd shapes, dtypes)
  profile/
    profile.sh              # Nsight Compute profiling script
    analyze_profile.py      # Script to parse and report profiler output
  benchmarks/
    benchmark.py            # Benchmark harness with JSON output
```

**Kernel README.md Structure**:
```markdown
# <Kernel Name>

## Problem Statement
[1-2 sentences describing the operation and its use cases]

## Intuition
[2-3 sentences: Explain the kernel conceptually before diving into GPU specifics. Use analogies if helpful.]

## GPU Execution Model
[1-2 paragraphs: How does the algorithm map to threads, blocks, warps? How does memory access pattern work? Include ASCII diagram if helpful.]

## Naive Implementation
[Link to or embed naive version. Explain why it's slow (e.g., uncoalesced access, low occupancy).]

## Optimized Implementation
[Step-by-step walkthrough of the optimized version. Explain each optimization:]
- Memory coalescing strategy
- Shared memory tiling (if applicable)
- Register usage and occupancy
- Thread-block layout
- Synchronization strategy

## Common Pitfalls
[3-4 bullets: Easy mistakes beginners make (uncoalesced access, shared memory bank conflicts, divergence, etc.) and how to detect/fix them.]

## How to Debug
[Concrete profiling commands to identify bottlenecks. Reference Nsight Compute metrics.]

## When to Use
[Applicability constraints: input size, GPU architecture, dtype, etc. When is the optimized version better than naive? When is cuBLAS better?]

## Benchmark Results
[Before/after performance with environment metadata (GPU model, driver, compiler flags).]

## References
[Cite relevant papers, tutorials, or canonical works that informed this implementation.]
```

**Acceptance Criteria**:
- [ ] Kernel compiles cleanly with recommended compiler flags (document flags in README).
- [ ] All correctness tests pass (run locally and in CI).
- [ ] Numerical accuracy within specified tolerance (absolute and relative).
- [ ] Benchmark harness runs and produces JSON output with environment metadata.
- [ ] Profiling script runs and generates Nsight Compute output (or equivalent).
- [ ] README follows pedagogical structure above.
- [ ] Code comments explain mapping to GPU concepts (threads, warps, memory hierarchy).
- [ ] At least 3 exercises (easy/medium/hard) provided for readers.

**Example Benchmark Output**:
```json
{
  "kernel": "vector_add",
  "dtype": "float32",
  "size": 10000000,
  "iterations": 100,
  "warmup_iterations": 10,
  "time_ms": {
    "mean": 2.45,
    "median": 2.43,
    "std": 0.08,
    "min": 2.38,
    "max": 2.71
  },
  "throughput_gb_s": 16.3,
  "gflops": 8.16,
  "environment": {
    "gpu": "NVIDIA A100 40GB",
    "driver": "535.104.05",
    "cuda_version": "12.2",
    "compute_capability": "8.0",
    "compiler": "nvcc 12.2",
    "compiler_flags": "-O3 -std=c++17",
    "timestamp": "2026-09-01T01:35:56Z"
  }
}
```

---

### 2. Adding an Optimization Recipe

**Scope**: Memory coalescing, shared memory tiling, loop unrolling, mixed precision, warp primitives, occupancy strategies, kernel fusion, streams, etc.

**Directory Structure**:
```
recipes/<category>/
  README.md                 # Recipe explanation
  example_*.cu              # 2-3 example implementations
  profile/
    compare.py              # Script comparing trade-offs with profiler metrics
```

**Recipe README.md Structure**:
```markdown
# <Optimization Technique>

## What & Why
[1-2 paragraphs: Explain the optimization technique and its impact (e.g., why coalescing matters for memory bandwidth).]

## When to Apply
[Criteria for when this optimization is beneficial. Roofline analysis: is the kernel memory-bound or compute-bound?]

## How It Works
[Technical explanation with diagrams or ASCII art if helpful.]

## Examples
[2-3 well-commented example kernels showing before/after.]

## Trade-offs
[Register pressure? Occupancy? Complexity? When does this optimization hurt?]

## How to Measure
[Which profiler metrics to examine? How to validate improvement?]

## Profiling Evidence
[Before/after comparison: memory throughput, cache hit rates, warp efficiency, occupancy, etc.]

## References
[Cite canonical papers and tutorials (Mark Harris, roofline model, etc.)]
```

**Acceptance Criteria**:
- [ ] README explains the technique clearly for beginners.
- [ ] At least 2 example implementations (before/after or alternative approaches).
- [ ] Profiling comparison script and data showing the trade-offs.
- [ ] References to authoritative sources.

---

### 3. Adding a Case Study

**Scope**: End-to-end optimization journey (naive → step 1 → step 2 → ... → best).

**Directory Structure**:
```
case_studies/<kernel>/
  README.md                 # Case study narrative
  step_0_naive/
    kernel.cu
    benchmark.py
  step_1_<optimization>/
    kernel.cu
    benchmark.py
  ...
  step_N_best/
    kernel.cu
    benchmark.py
    profile_*.txt          # Nsight Compute output
  exercises.md             # 3 exercises (easy/medium/hard)
  profiling_summary.md     # Analysis of before/after metrics
```

**Case Study README.md Structure**:
```markdown
# <Kernel> Optimization Case Study

## Problem
[Define the computation (e.g., GEMM: C += A @ B). Input shapes, accuracy requirements.]

## Optimization Journey
[Narrative: We'll optimize from naive to best by addressing bottlenecks step-by-step.]

### Step 0: Naive Implementation
[Performance baseline, profiling analysis, identified bottlenecks.]

### Step 1: [First Optimization]
[Explanation, code, benchmark improvement, profiling analysis.]

### Step 2: [Second Optimization]
...

### Step N: Best Configuration
[Final performance, roofline analysis, when to use this vs. cuBLAS/CUTLASS.]

## Key Insights
[3-5 bullets summarizing what we learned.]

## Profiling Evidence Summary
[Before/after table: throughput, occupancy, memory traffic, etc.]

## Exercises
[3 exercises for readers (see exercises.md).]

## References
```

**Exercises Template**:
```markdown
# Exercises

## Easy
1. [Modify naive kernel to reduce register usage by 10%. How does occupancy change?]

## Medium
2. [Implement a different thread-block tiling strategy. Compare performance and occupancy.]

## Hard
3. [Integrate autotuning to search tile sizes. What configuration does it find?]
```

**Acceptance Criteria**:
- [ ] All steps implemented and benchmarked.
- [ ] Profiling evidence (before/after Nsight Compute captures) included.
- [ ] Profiling analysis explains each optimization's impact.
- [ ] 3 exercises provided with difficulty levels.
- [ ] README narrative is clear and educational.

---

### 4. Adding Autotuning Integration

**Scope**: Parameterizable kernel with search harness and best-config saver.

**Directory Structure**:
```
autotune/<kernel>/
  kernel_parameterized.cu   # Kernel with tunable parameters
  tune.py                   # Autotuning search script
  best_config.json          # Saved best configuration
  README.md                 # Tuning guide and results
```

**Autotuning Script Template**:
```python
"""
Autotuning script for <kernel>.
- Defines search space (tile sizes, unroll factors, thread-block dims, etc.)
- Runs search (random, evolutionary, or guided)
- Saves best configuration
- Re-runs best config for validation
"""
import json
import subprocess
import random

# Define search space
SEARCH_SPACE = {
    "TILE_M": [32, 64, 128],
    "TILE_N": [32, 64, 128],
    "TILE_K": [8, 16, 32],
    "UNROLL": [1, 2, 4],
    "BLOCK_DIM_X": [16, 32],
    "BLOCK_DIM_Y": [16, 32],
}

def compile_and_benchmark(config):
    """Compile with config and return performance."""
    # Set NVCC flags based on config
    # Compile
    # Run benchmark
    # Return measured performance (GFLOPS, memory throughput, etc.)
    pass

def search():
    """Run autotuning search."""
    best_config = None
    best_perf = 0.0
    
    for trial in range(NUM_TRIALS):
        config = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        perf = compile_and_benchmark(config)
        
        if perf > best_perf:
            best_perf = perf
            best_config = config
        
        print(f"Trial {trial}: {config} -> {perf:.2f} GFLOPS")
    
    # Save best config
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    return best_config, best_perf

if __name__ == "__main__":
    best_config, best_perf = search()
    print(f"Best config: {best_config}")
    print(f"Best performance: {best_perf:.2f} GFLOPS")
```

**Acceptance Criteria**:
- [ ] Search space is clearly defined (tile sizes, block dims, unroll factors, etc.).
- [ ] Autotuning script runs and produces best_config.json.
- [ ] Script can re-run best config and validate performance.
- [ ] At least 100 trials completed (or documented constraint if fewer).
- [ ] README explains search strategy and best configuration discovered.
- [ ] Profiling comparison: best config vs. naive vs. cuBLAS/equivalent library.

---

### 5. Adding Profiling Integration

**Scope**: Nsight Compute scripts, metric parsers, and automated analysis.

**Acceptance Criteria**:
- [ ] Profiling script (bash or Python) runs Nsight Compute on your kernel.
- [ ] Parser extracts key metrics (occupancy, memory throughput, L1 hit rate, SM utilization, etc.).
- [ ] Generates a markdown or HTML report summarizing bottlenecks and optimization opportunities.
- [ ] Comparison script shows before/after metrics side-by-side.

---

## Code Style & Standards

### CUDA/C++

```cuda
// Clear, descriptive variable names
int tile_size = TILE_SIZE;  // Not 'ts' or 't'

// Comments explain GPU mapping
// Thread (bx, tx) loads element [by*TILE_SIZE + ty, bx*TILE_SIZE + tx]
// (coalesced access if tx varies, by varies with blocks)

// Avoid vendor-specific macros in code; centralize in a portability header
#ifdef CUDA_ENABLED
  cudaKernel<<<blocks, threads>>>(args);
#elif HIP_ENABLED
  hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, args);
#endif

// Prefer explicit error checking for clarity
cudaError_t err = cudaMemcpy(d_out, h_out, size, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    return;
}
```

### Python

```python
# Use type hints where helpful
def benchmark(kernel, input_size: int, iterations: int) -> dict:
    """Benchmark kernel and return metrics."""
    pass

# Comment non-obvious logic
# Parallel reduction: each warp reduces its portion independently,
# then a second kernel reduces across warps
```

### Documentation

- Use **bold** for GPU concepts (warp, block, shared memory) and code identifiers.
- Use numbered lists for procedures; bullet lists for options.
- Keep prose clear and accessible; avoid unnecessary jargon.

---

## Testing Requirements

1. **Correctness Tests**:
   - Compare against PyTorch reference or CPU baseline.
   - Test multiple input shapes (powers of 2, odd sizes, edge cases).
   - Test all supported dtypes (float32, float16, etc.).
   - Specify absolute and relative error tolerances.

2. **Smoke Tests**:
   - Build and run each kernel with small inputs.
   - No numerical verification needed; just ensure no crashes or NaNs.

3. **Benchmark Tests**:
   - Run with standard sizes and parameters.
   - Report JSON output with environment metadata.
   - Record at least 10 iterations (after warmup).

---

## Benchmarking & Profiling Requirements

### Benchmark Harness

```python
import json
import time
import torch

def benchmark_kernel(kernel_fn, size, dtype, iterations=100, warmup=10):
    """
    Benchmark kernel and return metrics.
    
    Args:
        kernel_fn: Callable that runs the kernel
        size: Input size (or shape tuple)
        dtype: torch dtype
        iterations: Number of timed runs
        warmup: Number of warmup runs
    
    Returns:
        dict with mean, median, std, min, max times (ms) and environment metadata
    """
    times = []
    
    # Warmup
    for _ in range(warmup):
        kernel_fn(size, dtype)
    
    # Timed runs
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        kernel_fn(size, dtype)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    time_per_iter = elapsed / iterations * 1000  # ms
    
    # Environment metadata
    return {
        "kernel": kernel_fn.__name__,
        "size": size,
        "dtype": str(dtype),
        "iterations": iterations,
        "time_ms": {
            "mean": time_per_iter,
            # ... median, std, min, max
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "driver": "...",  # Extracted from nvidia-smi or similar
            "cuda_version": torch.version.cuda,
            # ... other metadata
        }
    }

if __name__ == "__main__":
    results = benchmark_kernel(my_kernel, size=10_000_000, dtype=torch.float32)
    with open("benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
```

### Profiling with Nsight Compute

```bash
#!/bin/bash
# profile.sh: Profile kernel with Nsight Compute

KERNEL_EXECUTABLE="./build/kernel_benchmark"
METRICS="--metrics sm__utilization.avg,sm__warps_active.avg,memory_l1_system_coherence_requests_miss_rate"

ncu -f -o profile_output $METRICS $KERNEL_EXECUTABLE --size 10000000 --iterations 100

# Parse and report (see analyze_profile.py)
python analyze_profile.py profile_output.ncu-rep
```

---

## Pull Request Checklist

**Use this template when submitting a PR:**

```markdown
## Description
[Brief description of what this PR adds.]

## Type
- [ ] Foundational Kernel
- [ ] Optimization Recipe
- [ ] Case Study
- [ ] Autotuning Integration
- [ ] Profiling Tool
- [ ] Documentation
- [ ] Other: ___

## Acceptance Criteria Checklist

### Code Quality
- [ ] Code compiles cleanly with recommended flags
- [ ] No undefined behavior or memory issues
- [ ] Comments explain GPU mapping and key optimizations

### Correctness
- [ ] All unit tests pass
- [ ] Numerical tests pass with specified tolerances
- [ ] Edge cases handled (odd shapes, dtypes, boundary conditions)

### Performance & Profiling
- [ ] Benchmark harness runs and produces JSON output
- [ ] Environment metadata captured (GPU, driver, CUDA version, compiler flags)
- [ ] Profiler evidence included (Nsight Compute or equivalent)
- [ ] Before/after metrics analyzed (occupancy, throughput, cache hit rates, etc.)

### Documentation
- [ ] README follows pedagogical structure (Intuition, Minimal Example, Step-by-step, Pitfalls, Debugging, When to Use)
- [ ] Literature summary included (roofline, memory coalescing, warp primitives, etc.)
- [ ] All references cited (papers, tutorials, vendor docs)
- [ ] Code comments explain mapping to GPU concepts

### Reproducibility
- [ ] `reproduce.sh` script provided to rebuild and re-run benchmark
- [ ] Random seeds documented; all timings are deterministic (or distributions reported)
- [ ] Raw benchmark JSON and profiler outputs saved

### Testing
- [ ] Unit tests for correctness (CPU and GPU, if applicable)
- [ ] Smoke tests for build
- [ ] CI passes

## Performance Summary
[Before/after performance metrics in a table. Example:]

| Metric | Naive | Optimized | Improvement |
|--------|-------|-----------|-------------|
| GFLOPS | 50    | 250       | 5.0× |
| Memory BW (GB/s) | 30 | 90 | 3.0× |
| Occupancy (%) | 25 | 75 | +50pp |

## Environment
- GPU: [Model, e.g., NVIDIA A100 40GB]
- Driver: [e.g., 535.104.05]
- CUDA Version: [e.g., 12.2]
- Compiler: [e.g., nvcc 12.2, -O3 -std=c++17]

## References
- [Link to CUDA Programming Guide chapter]
- [Link to relevant paper or tutorial]
- [Cite canonical works: roofline model, Harris CUDA guides, etc.]
```

---

## Code Review Process

1. **Automated Checks**: All tests pass, code compiles, benchmarks run.
2. **Correctness Review**: Does the kernel compute the right result? Are tolerances reasonable?
3. **Performance Review**: Is the optimization real and significant? Is profiling evidence convincing?
4. **Documentation Review**: Can a beginner understand the kernel and optimization? Are trade-offs clear?
5. **Reproducibility Review**: Can we re-run the benchmark and profiling on different hardware?

---

## Attribution & Licensing

- KernelOpt is MIT-licensed. All contributions must be compatible with MIT licensing.
- Do not include proprietary code or vendor-licensed snippets without explicit attribution.
- Cite all papers, tutorials, and reference implementations you use.
- Include a `Co-authored-by` trailer in commit messages if multiple authors contribute.

---

## Questions?

Open an issue or discussion thread if you have questions. We're here to help!


# KernelOpt: Gold-Standard Kernel Engineering Learning & Optimization Library

**Transform GPU kernel engineering from art to science.**

KernelOpt is a comprehensive, research-grounded library of GPU kernel examples, optimization recipes, autotuning pipelines, and reproducible benchmarks that together teach kernel engineering and deliver practical, measurable performance improvements.

Every kernel in KernelOpt prioritizes **correctness first**, provides **step-by-step optimization narratives**, includes **rigorous profiling evidence**, and is grounded in **canonical GPU optimization theory** (roofline model, memory hierarchy, warp-level programming, occupancy analysis, etc.).

---

## Quick Start

**No GPU?** CPU reference tests work everywhere:
```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

**With CUDA?** Build and benchmark a kernel:
```bash
cd basics/vector_add
bash reproduce.sh        # Build, run, profile, and benchmark
cat benchmark.json       # View results
```

---

## Organization

### 📚 [Foundations](./basics/) — Minimal Working Kernels
Eight essential computations: vector add, dot product, matrix-vector multiply, GEMM, 1D/2D convolution, reduction, prefix-sum (scan).

Each kernel includes:
- **Naive + optimized implementations** (CUDA, PyTorch reference, Triton where applicable)
- **Pedagogical README** with intuition, GPU mapping, step-by-step optimization, pitfalls, debugging guide
- **Correctness tests** with multiple shapes and dtypes
- **Micro-benchmark harness** producing JSON output with environment metadata
- **Profiling setup** (Nsight Compute scripts, metric extraction)

**Start here:** [`basics/vector_add`](./basics/vector_add/)

### 🔧 [Optimization Recipes](./recipes/) — Reusable Techniques
Modular guides to 8+ core optimization techniques:

- **Memory**: Coalescing, shared memory tiling, avoiding bank conflicts, alignment, prefetching
- **Compute**: Loop unrolling, instruction-level parallelism, FMA usage, mixed precision, vectorization
- **Parallelism**: Thread/block/grid mapping, persistent threads, dynamic parallelism
- **Warp-Level**: Shuffles, ballot, reductions, warp-synchronous programming
- **Synchronization**: Efficient barriers, atomics, warp-aggregated atomics
- **Occupancy & Registers**: Balancing occupancy, register pressure, shared memory
- **Fusion & Splitting**: When to fuse kernels, when to split
- **Streams & I/O**: Async memcopy, overlapping compute and memory, host-device orchestration

Each recipe includes **multiple examples**, **trade-off analysis**, and **profiling comparison**.

### 🎯 [Case Studies](./case_studies/) — End-to-End Optimization Narratives
Three detailed optimization journeys: GEMM, 2D Convolution, and Reduction/Scan.

Each case study:
- Takes you from **naive → optimized** in 5–7 steps
- Explains **each bottleneck** using roofline analysis and profiler metrics
- Provides **annotated code** and **before/after benchmarks**
- Includes **3 exercises** (easy/medium/hard) for hands-on learning
- Shows **when to use hand-optimized vs. cuBLAS/CUTLASS**

**Start here:** [`case_studies/gemm`](./case_studies/gemm/)

### 🔬 [Autotuning & Compiler Optimization](./autotune/) — Automatic Parameter Search
Integrated autotuning for GEMM and convolution using TVM/AutoTVM or Kernel Tuner.

Each autotuning module:
- Defines a **parameterizable search space** (tile sizes, unroll factors, thread-block dims)
- Runs a **search harness** (random, evolutionary, or guided)
- Saves **best configuration** and validation results
- Includes **cost model and profiling integration**
- Produces **reproducible results** with environment metadata

### 📊 [Profiling & Analysis](./profiling/) — Understanding Bottlenecks
Reproducible profiling harnesses and automated analysis tools.

- **Nsight Compute integration**: Collect occupancy, memory throughput, cache hit rates, SM utilization, instruction mix
- **Metric extraction**: Parse profiler outputs and generate structured reports
- **Roofline plots**: Automatic roofline-style analysis showing memory vs. compute boundedness
- **Benchmark runner**: High-resolution timers, warmup/synchronization, JSON output with metadata

### 📖 [Documentation & Learning](./docs/) — Theoretical Grounding
Comprehensive guides for beginners and advanced practitioners.

- **[GPU Kernel Engineering Primer](./docs/GPU_PRIMER.md)**: Execution model, memory hierarchy, roofline model, occupancy, profiling basics with diagrams and analogies
- **[Literature & References](./docs/REFERENCES.md)**: 12 canonical topics (roofline, coalescing, warp programming, occupancy, autotuning, profiling, mixed precision, fusion, classic systems, GPU execution model, numerical stability, teaching resources) with 2–4 authoritative sources each
- **[Debugging Guide](./docs/DEBUGGING.md)**: How to detect and fix common kernel issues (divergence, bank conflicts, occupancy problems, etc.)
- **[Contributing Guidelines](./CONTRIBUTING.md)**: Submission checklist, code style, documentation standards, reproducibility requirements

---

## Learning Path

**Beginner (Week 1-2)**:
1. Read [GPU Kernel Engineering Primer](./docs/GPU_PRIMER.md)
2. Study [`basics/vector_add`](./basics/vector_add/) — understand coalescing and basic optimization
3. Read the **Memory Coalescing** recipe
4. Implement the vector_add exercises

**Intermediate (Week 3-4)**:
1. Study [`case_studies/gemm`](./case_studies/gemm/) — step through naive → optimized
2. Read **Occupancy & Register Pressure** recipe
3. Study GEMM's profiling evidence and do medium/hard exercises

**Advanced (Week 5-6)**:
1. Study autotuning in [`autotune/gemm`](./autotune/gemm/)
2. Read warp primitives and fusion recipes
3. Work through reduction/scan case study
4. Contribute a new kernel or recipe

---

## Scope & Completeness

### ✅ Included (Phase 1–4)

- **8 foundational kernels** with pedagogical documentation, tests, and profiling
- **8+ optimization recipes** with examples and trade-off analysis
- **3 end-to-end case studies** (GEMM, Conv2D, Reduction/Scan)
- **Autotuning integration** for GEMM and convolution
- **Profiling infrastructure** (Nsight Compute, metric extraction, analysis)
- **Comprehensive literature summary** (12 topics, 2–4 references each)

### 🔄 Future Phases

- **Tensor core optimization** for mixed-precision GEMM
- **Kernel fusion patterns** (MLP activation fusion, CNN+BN fusion)
- **SYCL and HIP ports** of foundational kernels
- **Performance portability** across NVIDIA, AMD, Intel GPUs
- **Distributed autotuning** (multi-GPU search)
- **Verified reference implementations** in Triton, JAX, PyTorch CUDA extensions

---

## Core Principles

1. **Correctness First**: Every result is numerically verified. Timings are meaningless if the result is wrong.
2. **Reproducibility**: All benchmarks include environment metadata (GPU, driver, CUDA version, compiler flags). Random seeds are documented. Results are deterministic or distributions are reported.
3. **Pedagogy**: Code comments explain the mapping from algorithm to GPU hardware. READMEs are tutorial-quality prose aimed at learners.
4. **References**: All optimizations are grounded in canonical works (roofline model, Harris CUDA guides, autotuning papers, etc.). References are cited and linked.
5. **Profiling Evidence**: Before/after metrics prove optimization claims. Bottleneck analysis using roofline and profiler metrics guides optimization priorities.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- Submission types and checklists (kernels, recipes, case studies, autotuning, profiling)
- Code style and documentation standards
- Testing and benchmarking requirements
- Pull request template

**Every contribution must include:**
- Correctness tests (numerical verification)
- Benchmark harness with JSON output
- Profiling evidence (before/after metrics)
- Pedagogical documentation
- Reproducibility script (`reproduce.sh`)

---

## Existing Kernels & Optimizations

KernelOpt builds on a solid foundation of existing optimized kernels:

| Kernel | Status | Main Optimization |
|--------|--------|-------------------|
| `vector_add` | ✓ Reference | Coalesced access, memory bandwidth |
| `row_softmax` | ✓ Reference | Online reduction, one block per row |
| `rms_norm` | ✓ Reference | Fused reduction + normalization |
| `fused_bias_gelu` | ✓ Reference | Elementwise fusion |
| `cnn_depthwise` | ✓ Reference | Channel-last tiling, reuse |
| `diffusion_timestep_embedding` | ✓ Reference | Fusion, launch overhead reduction |
| `rotary_embedding` | ✓ Reference | Vectorization |

These will be refactored and documented according to KernelOpt's pedagogical standards.

---

## Project Layout

```
KernelOpt/
├── basics/
│   ├── vector_add/          # Foundational kernels
│   ├── dot_product/
│   ├── matvec/
│   ├── gemm/
│   ├── conv1d/
│   ├── conv2d/
│   ├── reduction/
│   └── scan/
├── recipes/
│   ├── memory/              # Coalescing, shared memory, alignment
│   ├── compute/             # Unrolling, ILP, FMA, mixed precision
│   ├── parallelism/         # Thread/block mapping, persistent threads
│   ├── warp/                # Shuffles, reductions, warp-sync
│   ├── synchronization/     # Barriers, atomics
│   ├── occupancy/           # Occupancy strategies, register pressure
│   ├── fusion/              # Kernel fusion & splitting
│   └── io/                  # Streams, async memcopy
├── autotune/
│   ├── gemm/                # GEMM autotuning
│   └── conv2d/              # Conv2D autotuning
├── case_studies/
│   ├── gemm/                # GEMM: naive → best
│   ├── conv2d/              # Conv2D: naive → best
│   └── reduction_scan/      # Reduction/Scan: naive → best
├── profiling/
│   ├── nsight/              # Nsight Compute integration
│   └── analysis/            # Profiler output parsing & reporting
├── benchmarks/              # Shared benchmark utilities
├── tests/
│   ├── unit/                # Correctness tests
│   └── regression/          # Regression test suite
├── docs/
│   ├── GPU_PRIMER.md        # Learning guide
│   ├── REFERENCES.md        # Literature & references
│   ├── DEBUGGING.md         # Debugging guide
│   └── ...
├── kernelopt/               # Python library (legacy, to be refactored)
├── cuda/                    # Shared CUDA utilities (legacy)
├── CONTRIBUTING.md          # Contributor guidelines
└── README.md                # This file
```

---

## Method & Reproducibility

1. **Specify reference operation** and numerical tolerance.
2. **Check outputs** over awkward shapes, not only benchmark-friendly sizes.
3. **Warm up and synchronize** before measuring steady-state latency.
4. **Measure distribution**, not single timings (record mean, median, std, min, max).
5. **Record environment**: GPU model, driver, CUDA version, compiler flags, git revision.
6. **Save raw outputs**: JSON benchmark results and profiler captures.

---

## Requirements

- **CUDA**: 11.0+ (tested with 12.0+)
- **GPU**: Any CUDA-capable NVIDIA GPU (A100, H100, V100, RTX 4090, etc.)
- **Compiler**: GCC 7+, NVCC 11.0+
- **Python**: 3.9+ (for benchmarking and autotuning)
- **Optional**: PyTorch 1.13+, Triton 2.1+, Nsight Compute

CPU reference tests run everywhere. GPU tests require CUDA.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If KernelOpt helps your research or teaching, please cite:

```bibtex
@software{kernelopt2026,
  title = {KernelOpt: Gold-Standard Kernel Engineering Learning \& Optimization Library},
  author = {Contributors},
  year = {2026},
  url = {https://github.com/Praneeth-Suresh/KernelOpt}
}
```

---

## Acknowledgments

KernelOpt is grounded in decades of GPU optimization research and practice:
- Mark Harris's CUDA optimization guides and talks
- Roofline model (Williams et al., 2009)
- TVM/AutoTVM/Ansor papers and frameworks
- CUTLASS and cuBLAS design principles
- Academic resources (MIT, UC Berkeley, CMU)
- Community tutorials and best practices

See [docs/REFERENCES.md](./docs/REFERENCES.md) for full literature summary and citations.

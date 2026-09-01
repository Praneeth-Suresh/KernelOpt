# KernelOpt Implementation Roadmap

**Comprehensive roadmap for transforming KernelOpt into a gold-standard kernel engineering library.**

Last updated: 2026-09-01

---

## Overview

This roadmap outlines 36 distinct tasks organized into 6 phases, with dependencies, effort estimates, and success criteria for each task.

---

## Phase 0: Planning & Infrastructure ✅ **COMPLETE**

**Effort**: ~1 week | **Status**: ✅ Done | **PR**: Ready to merge

### Completed Tasks:
1. ✅ **plan-repo-struct**: Repository structure and top-level folder organization
2. ✅ **plan-literature-survey**: Literature & references document (12 topics, 40+ sources)
3. ✅ **plan-contrib-guidelines**: Contributing guidelines and PR template
4. ✅ **setup-ci-testing**: GitHub Actions CI pipeline (CPU tests, linting)

### Deliverables:
- Directory structure (`basics/`, `recipes/`, `autotune/`, `case_studies/`, `profiling/`, `docs/`)
- `docs/REFERENCES.md` (25KB, comprehensive literature grounding)
- `CONTRIBUTING.md` (18KB, submission guidelines)
- `.github/workflows/ci.yml` (GitHub Actions)
- Updated `README.md` (gateway documentation)

---

## Phase 1: Foundational Kernels — Part 1 ✅ **IN PROGRESS**

**Effort**: ~2 weeks | **Status**: Vector Add Done (1/8) | **Next**: Dot Product

### Kernel Implementation Order & Dependencies:

```
1. vector_add           ✅ Done    [1 day]
   ↓ (dependency)
2. dot_product          ⏳ Next    [1.5 days] — Introduces: Warp-level reduction, shared memory
3. matvec              📋 Queued  [1.5 days] — Introduces: 2D thread mapping, multiple loads per thread
4. gemm                📋 Queued  [2 days]   — Introduces: Tiling, register blocking, optimal occupancy
5. conv1d              📋 Queued  [1.5 days] — Introduces: Strided access patterns
6. conv2d              📋 Queued  [2 days]   — Introduces: 2D stencil, im2col, kernel fusion
7. reduction           📋 Queued  [1.5 days] — Introduces: Tree-based reduction, scan primitives
8. scan (prefix-sum)   📋 Queued  [1.5 days] — Introduces: Work-efficient algorithms, segmentation
```

### Completed (Vector Add):
- [x] README (pedagogical, 16KB)
- [x] CUDA kernel (naive + optimized + vectorized)
- [x] PyTorch reference
- [x] Correctness tests
- [x] Benchmark harness (JSON output)
- [x] Profiling setup (Nsight Compute)
- [x] Makefile + CMakeLists.txt
- [x] reproduce.sh script

### Task: basic-dot-product (Next)
**Effort**: 1.5 days | **Introduces**: Warp-level sync, shared memory, online reduction

**Deliverables**:
- README (warp-level primitives, reduction patterns)
- CUDA: naive (atomic reduction), optimized (warp-shuffle), shared memory blocking
- PyTorch ref + correctness tests + benchmarks + profiling
- 3 exercises (easy: modify tolerance; medium: alternative reduction strategy; hard: segmented reduction)

**Acceptance Criteria**:
- [ ] Tests pass (edge cases, numerical tolerance)
- [ ] Benchmark JSON saved
- [ ] Profiling script works
- [ ] README explains warp shuffles & online reduction
- [ ] Profiler shows improved occupancy vs. vector_add

---

## Phase 2: Optimization Recipes 📋

**Effort**: ~2 weeks | **Status**: Not started | **Depends on**: Phase 1 completion

### Recipe Categories (1–2 days each):

```
1. recipe-memory-coalescing
   - Coalesced vs. uncoalesced patterns
   - Multiple examples from vector_add, dot product
   - Profiling comparison (L1 hit rate, transaction count)
   
2. recipe-shared-memory-tiling
   - Tiling strategies, data layout optimization
   - Examples from GEMM, convolution
   - Shared memory bank conflicts (how to detect, avoid)
   
3. recipe-loop-unrolling
   - Trade-offs: register pressure vs. ILP
   - Compile-time vs. run-time unrolling
   - Profiling: instruction throughput impact
   
4. recipe-mixed-precision
   - FP16 vs. FP32 strategies
   - When to use each precision
   - Numerical stability checks
   
5. recipe-warp-primitives
   - Warp shuffles, ballot, reductions
   - Warp-synchronous algorithms
   - Comparison with __syncthreads()
   
6. recipe-occupancy-register-pressure
   - How registers/shared memory limit occupancy
   - Strategies to balance resources
   - Compile-time knob tuning
   
7. recipe-kernel-fusion
   - Producer-consumer locality
   - Register pressure vs. memory reuse tradeoffs
   - When fusion helps (data reuse) vs. hurts (occupancy)
   
8. recipe-streams-concurrency
   - Async memcopy, stream scheduling
   - Overlapping compute and H2D/D2H
   - Stream priorities and events
```

### Acceptance Criteria (per recipe):
- [ ] README explains technique clearly for beginners
- [ ] 2–3 example kernels (before/after or alternatives)
- [ ] Profiling comparison script showing tradeoffs
- [ ] References to authoritative sources

---

## Phase 3: Autotuning & Profiling Infrastructure 📋

**Effort**: ~1.5 weeks | **Status**: Not started | **Depends on**: Phase 1 + Phase 2

### Autotuning Tasks:

#### autotune-framework-integration
**Effort**: 1 day | **Choose**: TVM/AutoTVM or Kernel Tuner

- [ ] Install and integrate chosen framework
- [ ] Wrapper script for parameterized kernel compilation
- [ ] Basic search harness (random sampling or evolutionary)
- [ ] Config saver and re-run validation

#### autotune-gemm-tuner (depends on basic-gemm)
**Effort**: 1.5 days

**Search space**:
- Tile sizes: 32, 64, 128 (TILE_M, TILE_N, TILE_K)
- Unroll factors: 1, 2, 4
- Thread-block dims: various combinations
- Shared memory layout: row-major, column-major

**Deliverables**:
- Parameterized GEMM kernel
- Python search script (100+ trials)
- best_config.json saver
- Profiling integration (occupancy, memory BW vs. predicted)

#### autotune-conv2d-tuner (depends on basic-conv2d)
**Effort**: 1.5 days

Similar to GEMM but with convolution-specific parameters (stride, padding, dilation)

### Profiling Tasks:

#### profiling-nsight-setup
**Effort**: 1 day

- [ ] Script to run Nsight Compute
- [ ] Metric collection (occupancy, memory throughput, L1 hit rate, SM utilization, instruction mix)
- [ ] Error handling for missing tools

#### profiling-analysis-tools
**Effort**: 1.5 days

- [ ] Parser for Nsight Compute output
- [ ] Roofline plot generator
- [ ] Markdown report template
- [ ] Before/after comparison utility

---

## Phase 4: Case Studies 📋

**Effort**: ~1.5 weeks | **Status**: Not started | **Depends on**: Phase 1 + Phase 3

### case-gemm-end-to-end (depends on basic-gemm, autotune-gemm-tuner)
**Effort**: 3 days

**Optimization journey**:
1. Naive implementation (all threads read/write globally)
2. Tiled kernel (shared memory blocking for data reuse)
3. Register blocking (reduce global memory traffic further)
4. Vectorized variant (float4 loads, mixed precision)
5. Autotuned configuration (best parameters found by autotuner)
6. Comparison to cuBLAS reference

**Per step**:
- Annotated code with GPU mapping comments
- Benchmark (GFLOPS, memory throughput)
- Profiling snapshot (occupancy, L1 hit rate, instruction throughput)

**Final deliverables**:
- 6 kernel implementations (step 0–5)
- Before/after comparison table
- Roofline plot showing progression
- 3 exercises (easy: modify tile size; medium: alternative blocking strategy; hard: tensor cores)

### case-conv2d-end-to-end (depends on basic-conv2d, autotune-conv2d-tuner)
**Effort**: 2 days

**Optimization journey**:
1. Naive direct convolution
2. im2col + GEMM decomposition
3. Tiled direct convolution
4. Kernel fusion (bias + activation)
5. Autotuned configuration

### case-reduction-scan-end-to-end (depends on basic-reduction, basic-scan)
**Effort**: 2 days

**Optimization journey**:
1. Naive atomic-based reduction
2. Tree-based reduction (work-inefficient)
3. Work-efficient scan (Blelloch algorithm)
4. Segmented scan (for variable-length inputs)

---

## Phase 5: Pedagogy & Advanced Documentation 📋

**Effort**: ~1.5 weeks | **Status**: Not started | **Depends on**: Phase 1–4

### doc-gpu-primer
**Effort**: 1.5 days

**Contents** (`docs/GPU_PRIMER.md`):
- GPU memory hierarchy (diagram: registers → shared → L1/L2 → global)
- Execution model (threads, warps, blocks, SMs with timeline diagram)
- Roofline model (arithmetic intensity, memory vs. compute bound)
- Occupancy (how to calculate, impact on latency hiding)
- Profiling basics (key metrics, how to interpret)

**Format**: Beginner-friendly with analogies, diagrams (ASCII or SVG), worked examples

### doc-debugging-guide
**Effort**: 1 day

**Contents** (`docs/DEBUGGING.md`):
- Common issues: divergence, bank conflicts, occupancy bottleneck, numerical errors
- How to detect each (Nsight Compute metrics, Compute Sanitizer, printf debugging)
- Example kernels with bugs and fixes
- Profiling-driven debugging workflow

### doc-literature-summary
**Effort**: 0.5 days

- Review & augment `docs/REFERENCES.md`
- Ensure all 12 topics have 2–4 references
- Add "Key Takeaway for KernelOpt" notes

### doc-contributing-template
**Effort**: 0.5 days

- Finalize kernel submission template
- Add example PRs (vector_add as showcase)
- Checklist refinement based on experience

---

## Phase 6: Quality, Testing & Final Polish 📋

**Effort**: ~1.5 weeks | **Status**: Not started | **Depends on**: All phases

### test-unit-correctness
**Effort**: 1 day

- Add comprehensive tests for all kernels (edge cases, dtypes, shapes)
- Expand existing test suite (already started with vector_add)
- Define and document tolerances

### test-ci-integration
**Effort**: 1 day

- Extend CI pipeline to run smoke tests for all kernels
- GPU CI (self-hosted runner if available)
- Benchmark subsetting for quick CI runs

### repro-benchmark-runner
**Effort**: 1.5 days

**Deliverables**:
- Python utility to run benchmarks for all kernels
- Environment capture (GPU, driver, CUDA, compiler, git revision)
- JSON aggregation (all results in single file)
- CSV export for analysis

### repro-reproduce-scripts
**Effort**: 1 day

- Create top-level `reproduce.sh` (runs all kernels)
- Verify each kernel's `reproduce.sh` works independently
- Document expected outputs and how to interpret them

---

## Phase 7: Advanced Extensions (Future Work) 📋

**Not in current 9-week scope, but high-value for future**:

1. **Tensor Core Optimization** (FP16 matrix multiply with Tensor Cores)
2. **Kernel Fusion Patterns** (MLP activation, CNN+BN, etc.)
3. **SYCL & HIP Ports** (Multi-backend support)
4. **Performance Portability** (Autotuning across NVIDIA/AMD/Intel)
5. **Distributed Autotuning** (Multi-GPU search)
6. **Verified Reference Implementations** (Triton, JAX, PyTorch CUDA extensions)

---

## Dependency Graph

```
Phase 0 (Planning)
├─ Phase 1 (Foundational kernels)
│  ├─ Phase 2 (Recipes)
│  │  └─ Phase 3 (Autotuning)
│  │     ├─ Phase 4 (Case studies)
│  │     └─ Phase 5 (Pedagogy)
│  └─ Phase 3 (Autotuning)
│     └─ Phase 4 (Case studies)
└─ Phase 5 (Pedagogy)
└─ Phase 6 (Quality & reproducibility)
```

**Critical Path**: Phase 0 → Phase 1 (kernels) → Phase 3 (autotuning) → Phase 4 (case studies)

**Parallel Opportunities**:
- Phase 2 (recipes) can run in parallel with kernels
- Phase 5 (docs) can run alongside other phases
- Phase 6 (testing) should start mid-project

---

## Effort Estimate by Phase

| Phase | Tasks | Est. Days | Status |
|-------|-------|-----------|--------|
| 0 | 4 | 5 | ✅ Done |
| 1 | 8 | 11 | 🔄 1/8 done |
| 2 | 8 | 14 | 📋 Not started |
| 3 | 4 | 10 | 📋 Not started |
| 4 | 3 | 10 | 📋 Not started |
| 5 | 4 | 8 | 📋 Not started |
| 6 | 4 | 8 | 📋 Not started |
| **Total** | **35** | **66 days** | **~13 weeks** |

**Parallel work can reduce to ~9–10 weeks**

---

## Key Metrics & Success Criteria

### Quantitative:
- [ ] 8/8 foundational kernels implemented
- [ ] 8/8 optimization recipes with examples
- [ ] 3/3 end-to-end case studies
- [ ] 2/2 autotuning pipelines
- [ ] 100% test pass rate (CPU and GPU where available)
- [ ] All benchmarks produce JSON output with metadata
- [ ] All profiling scripts work without CUDA (graceful fallback)

### Qualitative:
- [ ] README documentation is "tutorial quality" (clear, engaging, beginner-friendly)
- [ ] All code includes GPU mapping comments
- [ ] All optimizations cite canonical sources (roofline, Harris, papers)
- [ ] Profiling evidence (before/after) included for all kernels
- [ ] Contributing guidelines followed by all submissions

### Educational:
- [ ] A beginner can understand vector_add in < 1 hour
- [ ] A developer can implement a new kernel following the template in < 1 day
- [ ] Researchers can use autotuning infrastructure for performance portability studies

---

## Known Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU access in CI | Use CPU tests; provide self-hosted runner template |
| Autotuning framework complexity | Start with Kernel Tuner (simpler than TVM) |
| Documentation maintenance burden | Template-driven docs; automated checks |
| Performance variability | Multiple runs, report distributions (mean/median/std) |
| Portability across GPU architectures | Keep kernel parameters architecture-agnostic; use compute_capability guards |

---

## Review Checkpoints

- **Week 1** (After Phase 0–1): Vector add PR merge; begin Phase 2
- **Week 3** (After Phase 1 complete): All 8 kernels implemented; begin Phase 3
- **Week 5** (After Phase 2–3): Recipes + autotuning complete; case studies started
- **Week 8** (After Phase 4–5): Case studies + pedagogy done; quality pass
- **Week 9** (Final): All tests green, documentation reviewed, ready for release

---

## Contributing Timeline Estimate

If multiple contributors work in parallel:

- **Contributor 1**: Kernels (Phase 1) → Case studies (Phase 4)
- **Contributor 2**: Recipes (Phase 2) → Pedagogy (Phase 5)
- **Contributor 3**: Autotuning (Phase 3) → Quality (Phase 6)

Expected time to completion with 3 full-time contributors: **4–5 weeks**

---

## Questions & Next Steps

1. **Ready to start Phase 2?** Pick the next kernel (dot product) and open a task/discussion
2. **Want to contribute?** Read `CONTRIBUTING.md` and pick a task from this roadmap
3. **Need clarification?** Open an issue

**Let's build the gold-standard kernel engineering library! 🚀**


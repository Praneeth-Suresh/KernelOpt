# KernelOpt

KernelOpt is a research-oriented collection of GPU kernels with an explanation for
each optimization decision. Every kernel has a readable PyTorch reference, a Triton
implementation, and (where useful) a CUDA C++ implementation. Correctness is a
precondition for timing: a result is never added to the catalogue merely because it
is fast.

The repository is designed for CUDA-capable Linux or WSL2 machines. The current
environment does not expose a CUDA device, so the checked-in catalogue intentionally
contains no fabricated timings. CPU reference tests and static checks remain useful
without a GPU.

## Kernels

| Kernel | Main optimization | Typical bottleneck |
| --- | --- | --- |
| `vector_add` | contiguous coalesced loads and one fused write | memory bandwidth |
| `row_softmax` | online max/sum reduction, one program per row | synchronization and memory traffic |
| `rms_norm` | fused reduction and normalization | memory bandwidth and launch overhead |
| `fused_bias_gelu` | fuses elementwise bias and GELU | intermediate traffic and launch overhead |

Each kernel's `docs/*.md` note explains the baseline, mapping to GPU execution,
memory behavior, numerical tradeoffs, and when the optimization is not appropriate.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

On a CUDA machine with PyTorch and Triton installed:

```bash
.venv/bin/python -m kernelopt.benchmark --kernel all --device cuda
```

Use `--jsonl results/run.jsonl` to save measurements. A benchmark refuses to claim a
GPU result when CUDA is unavailable. Add accepted evidence to the append-only
catalogue with:

```bash
.venv/bin/python -m kernelopt.catalogue append results/run.jsonl \
  --hardware "GPU model" --driver "driver version"
```

## Method

1. Specify a reference operation and tolerances.
2. Check outputs over awkward shapes, not only benchmark-friendly dimensions.
3. Warm up and synchronize before measuring steady-state latency.
4. Compare a distribution of repetitions, not a single timing.
5. Record software, hardware, shape, dtype, and git revision with the result.

The catalogue is append-only by policy. Historical evidence is not edited to improve
the narrative; new measurements supersede it only when their provenance is explicit.

## Project layout

```text
kernelopt/
  kernels/       references and Triton dispatch
  catalogue/     append-only evidence writer
  benchmark.py   correctness and timing harness
cuda/            optional CUDA C++ implementations
docs/            optimization principles and kernel notes
tests/           CPU-safe reference and API tests
results/         immutable-style checked-in catalogue records
```

## License

MIT. See [LICENSE](LICENSE).

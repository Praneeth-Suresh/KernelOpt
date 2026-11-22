# KernelOpt

Learn how to map simple tensor operations onto a GPU, starting with [vector addition](basics/vector_add/README.md). The worked example compares scalar and vectorized CUDA kernels, explains coalescing and tail handling, and includes reference tests and a benchmark driver.

## Start with vector addition

1. Read the [worked example](basics/vector_add/README.md) and the [short explanation of the memory mapping](docs/vector-add.md).
2. On Python 3.10 or newer, run the reference checks. A GPU is not required:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev]'
   python -m pytest tests/
   ```

3. On a CUDA-capable NVIDIA GPU with the matching toolkit, NVCC and CMake, follow the [vector-add benchmarking and profiling instructions](basics/vector_add/README.md#measure-on-your-gpu). The CPU checks are correctness checks, not GPU timing results.

## What is implemented

| Area | Where to start |
| --- | --- |
| Worked CUDA example | [Vector addition](basics/vector_add/) includes the kernel, build files, reference tests and profiling driver. |
| Python examples | [Kernel implementations](kernelopt/kernels/) include vector-add, row softmax, RMS norm, fused bias + GELU, depthwise convolution, diffusion timestep embedding and rotary embedding. Consult each implementation for its device-specific path. |
| GPU measurements | [Benchmark harness](kernelopt/benchmark.py) reports median and p95 latency. The [checked-in catalogue](results/catalogue.jsonl) records that the initial run had no CUDA device; it contains no measured speedup. |
| Next steps | [Roadmap](ROADMAP.md) and [historical project status](PROJECT_STATUS.md) distinguish completed work from proposed foundations, recipes and case studies. |

## Contributing

Include a reference operation, correctness checks on awkward as well as aligned shapes, and the raw benchmark output for any performance claim. Record the GPU, driver, CUDA version, input size and dtype, and git revision so the number can be reproduced. See [contributor guidelines](CONTRIBUTING.md), [optimization principles](docs/optimization-principles.md), and [references](docs/REFERENCES.md).

[MIT license](LICENSE).

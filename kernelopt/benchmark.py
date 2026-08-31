"""Correctness and synchronized benchmark harness."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .kernels import fused_bias_gelu, rms_norm, row_softmax, vector_add


@dataclass
class Measurement:
    kernel: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    median_us: float
    p95_us: float
    implementation: str = "kernelopt"


def _time(fn, warmup=25, repeats=100) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000)
    return values


def _case(name: str, device: torch.device) -> tuple[tuple[torch.Tensor, ...], callable]:
    dtype = torch.float16
    if name == "vector_add":
        args = (torch.randn(1 << 20, device=device, dtype=dtype),) * 2
        return args, lambda: vector_add(*args)
    if name == "row_softmax":
        args = (torch.randn(4096, 1024, device=device, dtype=dtype),)
        return args, lambda: row_softmax(args[0])
    if name == "rms_norm":
        x = torch.randn(4096, 1024, device=device, dtype=dtype)
        args = (x, torch.ones(1024, device=device, dtype=dtype))
        return args, lambda: rms_norm(*args)
    if name == "fused_bias_gelu":
        x = torch.randn(4096, 4096, device=device, dtype=dtype)
        args = (x, torch.randn(4096, device=device, dtype=dtype))
        return args, lambda: fused_bias_gelu(*args)
    raise ValueError(f"unknown kernel: {name}")


def benchmark(names: list[str], device: str) -> list[Measurement]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    target = torch.device(device)
    results = []
    for name in names:
        args, fn = _case(name, target)
        values = sorted(_time(fn))
        results.append(Measurement(name, tuple(args[0].shape), str(args[0].dtype), device,
                                   values[len(values) // 2], values[int(len(values) * 0.95)]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["all", "vector_add", "row_softmax", "rms_norm", "fused_bias_gelu"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    names = ["vector_add", "row_softmax", "rms_norm", "fused_bias_gelu"] if args.kernel == "all" else [args.kernel]
    results = benchmark(names, args.device)
    metadata = {"recorded_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "torch": torch.__version__, "git": _git_revision(), "results": [asdict(item) for item in results]}
    print(json.dumps(metadata, indent=2))
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("\n".join(json.dumps(asdict(item)) for item in results) + "\n")


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


if __name__ == "__main__":
    main()

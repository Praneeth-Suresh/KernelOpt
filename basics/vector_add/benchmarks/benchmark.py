#!/usr/bin/env python3
"""
Vector Add benchmark harness.

Produces JSON output with performance metrics and environment metadata.
Usage:
    python benchmark.py --size 100000000 --iterations 100 --output benchmark.json
"""
import json
import torch
import subprocess
import sys
import time
import numpy as np
from pathlib import Path


def get_cuda_info():
    """Extract CUDA environment info."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        
        # Try to get driver version
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info["driver_version"] = result.stdout.strip().split('\n')[0]
        except Exception:
            pass
    
    return info


def get_compiler_info():
    """Extract compiler info."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def benchmark_torch(size: int, iterations: int = 100, warmup: int = 10, device: str = "cuda") -> dict:
    """
    Benchmark PyTorch's native vector add.
    
    Args:
        size: Vector size
        iterations: Number of timed iterations
        warmup: Number of warmup iterations
        device: "cuda" or "cpu"
    
    Returns:
        Dictionary with timing results
    """
    a = torch.randn(size, dtype=torch.float32, device=device)
    b = torch.randn(size, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(warmup):
        c = a + b
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        c = a + b
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    # Verify correctness (first iteration)
    expected = a.cpu() + b.cpu()
    actual = c.cpu()
    max_error = (actual - expected).abs().max().item()
    
    # Throughput: 3 floats (A, B, C) × 4 bytes each = 12 bytes per element
    bytes_per_iter = size * 3 * 4
    bandwidth_gb_s = (bytes_per_iter / (times.mean() * 1e-3)) / 1e9
    
    return {
        "time_ms": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "std": float(times.std()),
            "min": float(times.min()),
            "max": float(times.max()),
            "all": times.tolist(),  # Raw data
        },
        "bandwidth_gb_s": float(bandwidth_gb_s),
        "gelements_per_s": float(size / (times.mean() * 1e-3) / 1e9),
        "max_numerical_error": float(max_error),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Add Benchmark")
    parser.add_argument("--size", type=int, default=100_000_000, help="Vector size")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--output", type=str, default="benchmark.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    print("Vector Add Benchmark")
    print("=" * 50)
    print(f"Vector size: {args.size:,} elements ({args.size * 4 / 1e6:.1f} MB per vector)")
    print(f"Iterations: {args.iterations}")
    print(f"Device: {args.device}")
    print()
    
    # Gather environment info
    cuda_info = get_cuda_info()
    compiler_info = get_compiler_info()
    
    print("Environment:")
    if cuda_info["cuda_available"]:
        print(f"  GPU: {cuda_info.get('gpu_name', 'unknown')}")
        print(f"  Compute Capability: {cuda_info.get('compute_capability', 'unknown')}")
        print(f"  Driver: {cuda_info.get('driver_version', 'unknown')}")
        print(f"  CUDA Version: {cuda_info.get('cuda_version', 'unknown')}")
    print(f"  Compiler: {compiler_info}")
    print()
    
    # Run benchmark
    print("Running benchmark...")
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            args.device = "cpu"
        
        results = benchmark_torch(args.size, iterations=args.iterations, warmup=args.warmup, device=args.device)
        
        print(f"Results:")
        print(f"  Mean time: {results['time_ms']['mean']:.3f} ms")
        print(f"  Median time: {results['time_ms']['median']:.3f} ms")
        print(f"  Std dev: {results['time_ms']['std']:.3f} ms")
        print(f"  Bandwidth: {results['bandwidth_gb_s']:.1f} GB/s")
        print(f"  Throughput: {results['gelements_per_s']:.3f} Gelements/s")
        print(f"  Max numerical error: {results['max_numerical_error']:.2e}")
        print()
        
        # Save JSON
        output = {
            "kernel": "vector_add",
            "backend": "pytorch",
            "device": args.device,
            "size": args.size,
            "iterations": args.iterations,
            "results": results,
            "environment": {
                "cuda": cuda_info,
                "compiler": compiler_info,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
        
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

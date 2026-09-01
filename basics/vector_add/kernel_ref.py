"""
Vector Add: PyTorch reference implementation.

This file provides a pure PyTorch CPU/GPU reference for correctness testing.
"""
import torch
import numpy as np


def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of vector add.
    
    Args:
        a: Input tensor of shape (N,)
        b: Input tensor of shape (N,)
    
    Returns:
        Output tensor c = a + b
    """
    return a + b


def benchmark_torch_native(size: int, iterations: int = 100, device: str = "cuda") -> dict:
    """
    Benchmark PyTorch's native vector add.
    
    Args:
        size: Vector size
        iterations: Number of iterations
        device: "cuda" or "cpu"
    
    Returns:
        Dictionary with timing and throughput metrics
    """
    a = torch.randn(size, dtype=torch.float32, device=device)
    b = torch.randn(size, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(10):
        c = vector_add_reference(a, b)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    import time
    times = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        c = vector_add_reference(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    # Throughput: 3 floats (A, B, C) × 4 bytes each = 12 bytes per element
    bytes_per_iter = size * 3 * 4
    bandwidth_gb_s = (bytes_per_iter / (times.mean() * 1e-3)) / 1e9
    
    return {
        "size": size,
        "iterations": iterations,
        "device": device,
        "time_ms": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "std": float(times.std()),
            "min": float(times.min()),
            "max": float(times.max()),
        },
        "bandwidth_gb_s": float(bandwidth_gb_s),
        "gflops": float(size / (times.mean() * 1e-3) / 1e9),  # Approximate
    }


def test_correctness(size: int, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    """
    Test correctness of vector add.
    
    Args:
        size: Vector size
        dtype: Data type
        device: "cuda" or "cpu"
    
    Raises:
        AssertionError if test fails
    """
    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    
    c = vector_add_reference(a, b)
    
    # Expected result (using CPU for high precision)
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c_cpu = a_cpu + b_cpu
    
    c_cpu_result = c.cpu().float()
    
    # Check relative error
    diff = (c_cpu_result - c_cpu).abs()
    rel_error = diff / (c_cpu.abs() + 1e-6)
    
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    print(f"Size: {size}, dtype: {dtype}, device: {device}")
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    
    # Allow higher tolerance for FP16
    tol = 1e-2 if dtype == torch.float16 else 1e-5
    assert max_rel_error < tol, f"Error too large: {max_rel_error} > {tol}"
    print("PASS\n")


if __name__ == "__main__":
    # Test correctness on various sizes and dtypes
    print("=== Correctness Tests ===\n")
    
    for size in [3, 256, 1000000]:
        for dtype in [torch.float32]:
            test_correctness(size, dtype, "cpu")
    
    # Benchmark PyTorch's native implementation
    print("\n=== Benchmarks ===\n")
    
    try:
        if torch.cuda.is_available():
            results = benchmark_torch_native(100000000, iterations=100, device="cuda")
            print(f"PyTorch CUDA: {results['time_ms']['mean']:.2f} ms")
            print(f"  Bandwidth: {results['bandwidth_gb_s']:.1f} GB/s\n")
    except Exception as e:
        print(f"CUDA benchmark failed: {e}\n")
    
    results = benchmark_torch_native(100000000, iterations=100, device="cpu")
    print(f"PyTorch CPU: {results['time_ms']['mean']:.2f} ms")
    print(f"  Bandwidth: {results['bandwidth_gb_s']:.1f} GB/s\n")

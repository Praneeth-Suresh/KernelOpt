"""Coalesced elementwise vector addition."""

import torch


def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def _triton_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    import triton
    import triton.language as tl

    out = torch.empty_like(a)
    n = out.numel()

    @triton.jit
    def kernel(x_ptr, y_ptr, o_ptr, size, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(o_ptr + offsets, x + y, mask=mask)

    kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK=256)
    return out


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add tensors, dispatching to Triton for CUDA and PyTorch otherwise."""
    if a.device.type != "cuda":
        return vector_add_reference(a, b)
    return _triton_vector_add(a, b)

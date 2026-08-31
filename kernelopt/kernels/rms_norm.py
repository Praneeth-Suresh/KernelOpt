"""Fused RMS normalization."""

import torch


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(x.dtype)) * weight


def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import triton
    import triton.language as tl

    rows, cols = x.shape
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, w_ptr, o_ptr, cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < cols
        values = tl.load(x_ptr + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(values * values, axis=0) / cols + eps)
        weights = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        tl.store(o_ptr + row * cols + offsets, (values * scale * weights).to(x_ptr.dtype.element_ty), mask=mask)

    block = min(1 << (cols - 1).bit_length(), 4096)
    kernel[(rows,)](x, weight, out, cols, eps, BLOCK=block)
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.device.type != "cuda":
        return rms_norm_reference(x, weight, eps)
    if x.ndim != 2 or weight.shape != (x.shape[-1],):
        raise ValueError("rms_norm expects [rows, hidden] and [hidden] weight")
    return _triton_rms_norm(x, weight, eps)

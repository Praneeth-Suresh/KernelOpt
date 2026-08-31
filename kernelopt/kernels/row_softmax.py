"""Numerically stable row-wise softmax with one Triton program per row."""

import torch


def row_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _triton_row_softmax(x: torch.Tensor) -> torch.Tensor:
    import triton
    import triton.language as tl

    rows, cols = x.shape
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, o_ptr, cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < cols
        values = tl.load(x_ptr + row * cols + offsets, mask=mask, other=-float("inf"))
        maximum = tl.max(values, axis=0)
        exponents = tl.exp(values - maximum)
        denominator = tl.sum(tl.where(mask, exponents, 0.0), axis=0)
        tl.store(o_ptr + row * cols + offsets, exponents / denominator, mask=mask)

    block = min(1 << (cols - 1).bit_length(), 4096)
    kernel[(rows,)](x, out, cols, BLOCK=block)
    return out


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":
        return row_softmax_reference(x)
    if x.ndim != 2:
        raise ValueError("row_softmax expects a 2D tensor")
    return _triton_row_softmax(x)

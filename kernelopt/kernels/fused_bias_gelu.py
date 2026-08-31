"""Fused bias addition and GELU activation."""

import torch
import torch.nn.functional as F


def fused_bias_gelu_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(x + bias)


def _triton_fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    import triton
    import triton.language as tl

    out = torch.empty_like(x)
    n = x.numel()
    width = x.shape[-1]

    @triton.jit
    def kernel(x_ptr, b_ptr, o_ptr, n, width, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        biases = tl.load(b_ptr + offsets % width)
        values = values + biases
        c = 0.7978845608028654
        result = 0.5 * values * (1.0 + tl.tanh(c * (values + 0.044715 * values * values * values)))
        tl.store(o_ptr + offsets, result, mask=mask)

    kernel[(triton.cdiv(n, 256),)](x, bias, out, n, width, BLOCK=256)
    return out


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":
        return fused_bias_gelu_reference(x, bias)
    if x.ndim < 1 or bias.shape != (x.shape[-1],):
        raise ValueError("bias must match the last dimension")
    return _triton_fused_bias_gelu(x, bias)

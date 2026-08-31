"""Fused rotary positional embedding for transformer query/key tensors."""

import torch


def rotary_embedding_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return rotated.flatten(-2)


def _triton_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    import triton
    import triton.language as tl

    batch, sequence, heads, dimension = x.shape
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, c_ptr, s_ptr, o_ptr, sequence, heads, dimension, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        pair = tl.arange(0, BLOCK)
        half = dimension // 2
        valid = pair < half
        token = pid // heads
        head = pid % heads
        base = (token * heads + head) * dimension
        even = tl.load(x_ptr + base + 2 * pair, mask=valid, other=0.0)
        odd = tl.load(x_ptr + base + 2 * pair + 1, mask=valid, other=0.0)
        cosine = tl.load(c_ptr + token * half + pair, mask=valid, other=1.0)
        sine = tl.load(s_ptr + token * half + pair, mask=valid, other=0.0)
        tl.store(o_ptr + base + 2 * pair, even * cosine - odd * sine, mask=valid)
        tl.store(o_ptr + base + 2 * pair + 1, even * sine + odd * cosine, mask=valid)

    block = min(1 << ((dimension // 2) - 1).bit_length(), 1024)
    kernel[(batch * sequence * heads,)](x, cos, sin, out, sequence, heads, dimension, BLOCK=block)
    return out


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs; cos/sin have shape [sequence, dimension / 2]."""
    if x.device.type != "cuda":
        return rotary_embedding_reference(x, cos, sin)
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("x must be [batch, sequence, heads, even_dimension]")
    expected = (x.shape[1], x.shape[-1] // 2)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError("cos and sin must be [sequence, dimension / 2]")
    return _triton_rotary_embedding(x, cos, sin)

"""Fused sinusoidal timestep embedding used by diffusion U-Nets."""

import math
import torch


def diffusion_timestep_embedding_reference(
    timesteps: torch.Tensor, dimension: int, max_period: float = 10000.0
) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    result = torch.cat([angles.cos(), angles.sin()], dim=-1)
    return torch.cat([result, torch.zeros_like(result[:, :1])], dim=-1) if dimension % 2 else result


def _triton_diffusion_embedding(
    timesteps: torch.Tensor, dimension: int, max_period: float
) -> torch.Tensor:
    import triton
    import triton.language as tl

    batch = timesteps.numel()
    half = dimension // 2
    out = torch.empty((batch, dimension), device=timesteps.device, dtype=torch.float32)

    @triton.jit
    def kernel(t_ptr, o_ptr, half, dimension, log_period, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < half
        frequency = tl.exp(-log_period * offsets / half)
        angle = tl.load(t_ptr + row).to(tl.float32) * frequency
        tl.store(o_ptr + row * dimension + offsets, tl.cos(angle), mask=mask)
        tl.store(o_ptr + row * dimension + half + offsets, tl.sin(angle), mask=mask)
        if dimension % 2:
            tl.store(o_ptr + row * dimension + dimension - 1, 0.0)

    block = min(1 << (half - 1).bit_length(), 1024)
    kernel[(batch,)](timesteps, out, half, dimension, math.log(max_period), BLOCK=block)
    return out


def diffusion_timestep_embedding(
    timesteps: torch.Tensor, dimension: int, max_period: float = 10000.0
) -> torch.Tensor:
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if timesteps.device.type != "cuda":
        return diffusion_timestep_embedding_reference(timesteps, dimension, max_period)
    return _triton_diffusion_embedding(timesteps, dimension, max_period)

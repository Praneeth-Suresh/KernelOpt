"""Channel-last depthwise 3x3 convolution fused with bias and ReLU."""

import torch
import torch.nn.functional as F


def cnn_depthwise_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Reference NCHW operation, with the public API expressed as NHWC."""
    nchw = x.permute(0, 3, 1, 2)
    result = F.conv2d(nchw, weight[:, None], bias=bias, padding=1, groups=x.shape[-1])
    return result.permute(0, 2, 3, 1).relu()


def _triton_cnn_depthwise(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    import triton
    import triton.language as tl

    batch, height, width, channels = x.shape
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, w_ptr, b_ptr, o_ptr, height, width, channels,
               BLOCK_C: tl.constexpr):
        pid = tl.program_id(0)
        channel = pid % channels
        pixel = pid // channels
        batch_id = pixel // (height * width)
        remainder = pixel % (height * width)
        row = remainder // width
        col = remainder % width
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < 1
        acc = tl.zeros((1,), dtype=tl.float32)
        for kh in range(3):
            for kw in range(3):
                rr = row + kh - 1
                cc = col + kw - 1
                valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                value = tl.load(
                    x_ptr + ((batch_id * height + rr) * width + cc) * channels + channel,
                    mask=valid, other=0.0,
                )
                kernel_value = tl.load(w_ptr + channel * 9 + kh * 3 + kw)
                acc += value * kernel_value
        acc += tl.load(b_ptr + channel)
        tl.store(o_ptr + ((batch_id * height + row) * width + col) * channels + channel,
                 tl.maximum(acc, 0.0), mask=mask)

    kernel[(batch * height * width * channels,)](x, weight, bias, out, height, width,
                                                  channels, BLOCK_C=1)
    return out


def cnn_depthwise(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply NHWC depthwise 3x3 convolution, bias, and ReLU."""
    if x.device.type != "cuda":
        return cnn_depthwise_reference(x, weight, bias)
    if x.ndim != 4 or weight.shape != (x.shape[-1], 3, 3):
        raise ValueError("expected x=[N,H,W,C] and weight=[C,3,3]")
    if bias.shape != (x.shape[-1],):
        raise ValueError("bias must match channels")
    return _triton_cnn_depthwise(x, weight, bias)

# RMS normalization

RMSNorm rescales each row according to its root-mean-square magnitude:
`y = x / sqrt(mean(x²) + eps) * weight`. It is common in modern Transformer blocks.
Unlike LayerNorm, it does not subtract the row mean; that removes one reduction and
is part of the operation's definition.

The kernel first reduces the squared values, then immediately uses the resulting
scale to produce the output. The important optimization is fusion: the unnormalized
row never becomes a temporary tensor in device memory. Inputs may be FP16 or BF16,
but the sum is accumulated in FP32 so rounding errors do not dominate the scale.

One program per row avoids cross-program synchronization and works well when the
hidden dimension fits a practical block. Power-of-two block sizes make masking and
compiler code generation straightforward, but they add unused lanes for unusual
dimensions. Extremely wide rows can cause register pressure; a tuned implementation
may split the reduction across programs and combine partial sums.

## Further reading

* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) introduces
  RMSNorm and compares it with LayerNorm.
* [Triton layer normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
  illustrates fused row reductions and affine scaling.
* [NVIDIA mixed-precision training guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
  discusses why FP32 accumulation matters for low-precision workloads.

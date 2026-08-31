# Diffusion timestep embedding

Diffusion U-Nets condition every denoising step on a scalar timestep. The standard
embedding evaluates log-spaced frequencies followed by sine and cosine. A staged
implementation may materialize frequencies and phase values; the fused kernel
computes the phase per output lane and writes the two trigonometric halves directly.

The operation is small relative to a U-Net convolution, so launch latency and
temporary allocation matter more than peak arithmetic throughput. Frequencies are
generated in FP32 to preserve phase accuracy. This kernel is most useful when many
small denoising requests are batched; for a large batch, a library vector operation
may be equally effective. Odd embedding dimensions are padded with zero.

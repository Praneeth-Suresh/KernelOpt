# Diffusion timestep embedding

A diffusion model repeatedly denoises an image or latent representation. At each
step, a U-Net needs to know *which* step it is performing. The input is a scalar
timestep, but a neural network benefits from a richer signal, so the scalar is
expanded into sine and cosine values at logarithmically spaced frequencies.

KernelOpt computes those frequencies and phases inside the kernel. Each program
handles one timestep and writes the cosine half and sine half directly. A staged
version might first create a frequency vector, then a phase matrix, then the
trigonometric results. The fused version avoids those temporary allocations and
their associated memory traffic. FP32 phase arithmetic helps preserve accuracy
when a large timestep is multiplied by a small frequency.

This embedding is small beside a U-Net convolution, which makes launch latency
noticeable for small batches. For large batches, a highly optimized library
operation may match or exceed this implementation. Odd output dimensions are
defined by padding the final feature with zero.

## Further reading

* [DDPM](https://arxiv.org/abs/2006.11239) introduces the denoising diffusion
  probabilistic model and its timestep-conditioned denoiser.
* [Improved DDPM](https://arxiv.org/abs/2102.09672) discusses practical diffusion
  improvements and timestep parameterizations.
* [OpenAI guided-diffusion timestep embedding](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py)
  contains a readable reference implementation of sinusoidal embeddings.

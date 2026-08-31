# Channel-last depthwise convolution

In a normal convolution, every output channel mixes many input channels. A
**depthwise** convolution is different: channel 0 uses only channel 0, channel 1
uses only channel 1, and so on. Each channel therefore receives its own small
spatial filter. This saves arithmetic, but it also leaves less cross-channel reuse
for the GPU to exploit.

The KernelOpt example accepts NHWC data: batch, height, width, channels. Adjacent
channels at one pixel are adjacent in memory, so a group of lanes can load them
efficiently. Each program computes one output pixel and channel, visits its 3x3
neighborhood, adds the channel bias, and applies ReLU before writing the result.
Border predicates treat positions outside the image as zero; no separately padded
image is allocated.

Bias and ReLU are fused with the convolution, removing intermediate stores, loads,
and a launch. The tradeoff is important: a one-output mapping does not reuse a
filter across many output pixels as aggressively as a tuned dense-convolution
library. NHWC is also not free if the surrounding CNN is NCHW, because layout
conversion can consume the saved time.

## Further reading

* [Depthwise Separable Convolutions](https://arxiv.org/abs/1704.04861) explains why
  separating spatial filtering from channel mixing reduces computation.
* [PyTorch depthwise convolution documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  defines the `groups` interpretation used by the reference.
* [CUDA Best Practices: Memory Layout](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#device-memory-spaces)
  discusses the access patterns that make a tensor layout GPU-friendly.

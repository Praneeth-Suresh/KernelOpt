# Channel-last depthwise convolution

Depthwise convolution applies an independent spatial filter to each channel. Unlike
dense convolution, it has little cross-channel reuse and is commonly limited by
loads, stores, and address calculation. This kernel uses NHWC layout so adjacent
threads can read adjacent channels at a pixel, fuses bias and ReLU, and maps one
output element to one program. Border predicates implement zero padding without a
separate padded tensor.

The fusion removes two intermediate global-memory operations and one launch. The
tradeoff is that a one-program-per-output mapping does not exploit the high reuse
available in dense convolutions; vendor implicit-GEMM libraries are usually preferable
for dense or very large channel-mixing layers. Layout conversion can also erase the
benefit if the surrounding CNN is NCHW.

# RMS normalization

RMS normalization needs the mean square of a row, a reciprocal square root, and a
scale by learned weights. The Triton kernel fuses the reduction and output transform,
so the input is read once and no unnormalized intermediate is written to global
memory. The sum is accumulated in FP32 to reduce error for low-precision inputs.

One program per row is appropriate when the hidden dimension fits a block. Register
pressure and the power-of-two padding used for masked loads become costs for very
large hidden dimensions. Layer normalization is not interchangeable: it also needs a
mean reduction and therefore a different numerical and synchronization design.

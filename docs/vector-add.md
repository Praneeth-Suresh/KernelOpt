# Vector add

Vector addition performs one read from each input and one write for each element.
There is little arithmetic to hide the cost of memory traffic, so the relevant
optimization is a contiguous, coalesced access pattern with no unnecessary temporary.
The Triton implementation assigns a fixed block of adjacent offsets to each program
and masks the final partial block. The CUDA counterpart assigns one adjacent element
to each thread.

This operation is generally bandwidth-bound. Larger blocks can reduce scheduling
overhead, but excessive blocks do not increase reuse because every element is read
once. Alignment, strides, and competing traffic can dominate the result; the kernel
is intended for contiguous tensors.

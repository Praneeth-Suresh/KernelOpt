# Vector add

Vector addition is the simplest useful GPU example: for every position `i`, compute
`out[i] = a[i] + b[i]`. The addition itself is cheap. Each element still requires
two reads and one write from device memory, so the time is usually determined by
how efficiently those bytes travel.

The Triton kernel gives each program a consecutive block of positions. Consecutive
lanes therefore request consecutive addresses, allowing the GPU to coalesce memory
transactions. The mask on the final block is important: a vector of length 1000
does not divide evenly into blocks, and unmasked lanes would read or write beyond
the tensor. The CUDA version makes the same mapping explicit with one element per
thread.

There is no reuse to gain from tiling: each value is consumed once. A larger block
may reduce scheduling overhead, but eventually increases register use without
reducing memory traffic. Contiguous storage is an assumption of this example;
strided inputs may require a different mapping.

## Further reading

* [CUDA Best Practices: Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access)
  explains why adjacent lanes should access adjacent memory.
* [Triton vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
  walks through the same blocked-and-masked pattern.

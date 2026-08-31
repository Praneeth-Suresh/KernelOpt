# Fused bias GELU

An MLP commonly computes `GELU(x + bias)`: add a learned bias to every feature, then
apply the smooth GELU activation. Written as two ordinary tensor operations, the
bias result may be stored in device memory and loaded again by GELU. The arithmetic
is modest, but the extra round trip and kernel launch are real costs.

KernelOpt performs both stages in one program. It loads an input value and its
per-column bias, evaluates the tanh approximation to GELU while the value is still
in a register, and stores only the final result. The mapping is elementwise, so
neighboring lanes can coalesce their accesses. Broadcasting the bias across rows
means one compact parameter vector serves the whole activation matrix.

Fusion is not automatically faster. The combined expression uses more instructions
and registers, which can lower occupancy. Compilers and vendor libraries may already
fuse this pattern, and replacing them without measuring can regress performance.
The useful question is whether the fusion removes a materialized intermediate on the
actual model shape.

## Further reading

* [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) defines the
  activation and its relationship to stochastic regularization.
* [PyTorch GELU documentation](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
  documents the exact and tanh-approximate forms.
* [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
  provides a concrete example of eliminating intermediate tensors through fusion.

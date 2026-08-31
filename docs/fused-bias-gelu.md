# Fused bias GELU

The expression `GELU(x + bias)` is two elementwise stages if written directly:
the bias result is stored and loaded before activation. The fused kernel reads `x`
and the per-column bias, evaluates the tanh GELU approximation in registers, and
writes one result. This removes an intermediate global-memory round trip and one
launch.

Fusion increases instruction count and register use per thread. It is beneficial
when the unfused intermediate is material and the operation is not already fused by
the surrounding compiler. It should not replace a vendor library path without
benchmarking the actual model shape and dtype.

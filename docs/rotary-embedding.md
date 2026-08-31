# Rotary positional embedding

Self-attention needs information about order: without it, swapping two tokens would
not change the set of vectors presented to the attention operation. Rotary position
embedding (RoPE) supplies that information by rotating each adjacent pair of query
or key features by a position-dependent angle.

For a pair `(x_even, x_odd)`, the output is
`(x_even cos(theta) - x_odd sin(theta), x_even sin(theta) + x_odd cos(theta))`.
A straightforward tensor implementation may create stacked and reshaped
intermediates. KernelOpt maps one program to a token-head, loads each pair once,
applies the two-by-two rotation in registers, and writes the final layout directly.

The cosine and sine tables are shared across heads and read-only, so the same
position data can be reused by every head. The implementation assumes contiguous
`[batch, sequence, heads, dimension]` tensors and an even head dimension. It does
not calculate the attention scores. Keeping the rotation separate makes correctness
easy to test; a production attention kernel may fuse rotation into Q/K loading if
the resulting register use remains acceptable.

## Further reading

* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  introduces RoPE and derives its relative-position behavior.
* [Hugging Face rotary embedding utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
  shows how RoPE is integrated into a production Transformer.
* [FlashAttention-2](https://arxiv.org/abs/2307.08691) discusses fused attention
  kernels and the tradeoffs of keeping more work inside one GPU program.

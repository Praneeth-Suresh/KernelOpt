# Rotary positional embedding

Rotary position embedding (RoPE) applies a position-dependent 2D rotation to adjacent
query/key features. The operation is elementwise over feature pairs, but a naive
implementation creates stacked, rotated, and reshaped intermediates. KernelOpt maps
one program to each token-head and loads each pair once, applying the rotation and
writing the final layout directly.

The cosine and sine tables are shared by all heads, so they are read-only broadcast
inputs. The kernel assumes contiguous `[batch, sequence, heads, dimension]` tensors
and an even head dimension. It does not compute the attention score matrix; keeping
RoPE separate makes the numerical boundary explicit, while a production attention
kernel may fuse it with Q/K loading when register pressure permits.

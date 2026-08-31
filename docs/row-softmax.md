# Row-wise softmax

Softmax turns a row of scores into positive values that sum to one:
`softmax(x_i) = exp(x_i) / sum_j exp(x_j)`. Attention uses this operation constantly,
but the formula hides a numerical hazard: exponentiating a large score can overflow.
The stable form subtracts the row maximum before taking exponentials. This changes
neither the mathematical result nor the normalized probabilities.

A simple implementation may launch separate kernels for the maximum, exponentials,
and sum, materializing temporary arrays between them. KernelOpt assigns one Triton
program to each row. That program loads the row, computes its maximum, computes the
exponentials and their sum, and writes the normalized values. Because one program
owns the entire row, no other program must synchronize to obtain the denominator.
Masked lanes represent columns beyond a short row and contribute zero to the sum.

The design keeps row values and reduction state close to the program, reducing global
traffic. It is a good fit for moderate row widths. Very wide rows can consume too
many registers, while very short rows may spend more time launching than computing.
Production attention implementations often use specialized fused attention kernels
when softmax is only one stage of a larger computation.

## Further reading

* [Online Softmax](https://arxiv.org/abs/1805.02867) explains stable streaming
  reductions for softmax.
* [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
  demonstrates the one-program-per-row design.
* [FlashAttention](https://arxiv.org/abs/2205.14135) shows how tiling and fusion
  can avoid materializing the full attention matrix.

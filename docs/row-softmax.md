# Row-wise softmax

Naive softmax often materializes a maximum, exponentials, and a sum as separate
operations. KernelOpt assigns one program to each row and performs the maximum,
exponential, and sum in one pass over the row. Subtracting the maximum prevents
overflow, while FP32 reduction behavior is delegated to the Triton type rules.

The one-program mapping avoids inter-program synchronization because a row's
normalization denominator is local. It is effective for moderate row widths. Very
wide rows can exceed register capacity, and very short rows can be dominated by
launch overhead; a production dispatcher should benchmark shape-specific alternatives.

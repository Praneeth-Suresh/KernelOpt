import pytest
import torch

from kernelopt import fused_bias_gelu, rms_norm, row_softmax, vector_add
from kernelopt.kernels.fused_bias_gelu import fused_bias_gelu_reference
from kernelopt.kernels.rms_norm import rms_norm_reference
from kernelopt.kernels.row_softmax import row_softmax_reference


def test_vector_add_matches_reference():
    a, b = torch.randn(33), torch.randn(33)
    torch.testing.assert_close(vector_add(a, b), a + b)


@pytest.mark.parametrize("shape", [(3, 7), (9, 128)])
def test_row_softmax_matches_reference(shape):
    x = torch.randn(*shape)
    torch.testing.assert_close(row_softmax(x), row_softmax_reference(x))


def test_rms_norm_matches_reference():
    x, weight = torch.randn(5, 17), torch.randn(17)
    torch.testing.assert_close(rms_norm(x, weight), rms_norm_reference(x, weight), rtol=1e-5, atol=1e-6)


def test_fused_bias_gelu_matches_reference():
    x, bias = torch.randn(4, 11), torch.randn(11)
    torch.testing.assert_close(fused_bias_gelu(x, bias), fused_bias_gelu_reference(x, bias))

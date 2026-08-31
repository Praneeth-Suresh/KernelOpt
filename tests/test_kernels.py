import pytest
import torch

from kernelopt import (
    cnn_depthwise,
    diffusion_timestep_embedding,
    fused_bias_gelu,
    rms_norm,
    rotary_embedding,
    row_softmax,
    vector_add,
)
from kernelopt.kernels.cnn_depthwise import cnn_depthwise_reference
from kernelopt.kernels.diffusion_embedding import diffusion_timestep_embedding_reference
from kernelopt.kernels.fused_bias_gelu import fused_bias_gelu_reference
from kernelopt.kernels.rms_norm import rms_norm_reference
from kernelopt.kernels.row_softmax import row_softmax_reference
from kernelopt.kernels.rope import rotary_embedding_reference


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


def test_cnn_depthwise_matches_reference():
    x = torch.randn(2, 5, 6, 3)
    weight = torch.randn(3, 3, 3)
    bias = torch.randn(3)
    torch.testing.assert_close(cnn_depthwise(x, weight, bias), cnn_depthwise_reference(x, weight, bias))


def test_diffusion_embedding_matches_reference():
    timesteps = torch.tensor([0, 1, 1000])
    torch.testing.assert_close(
        diffusion_timestep_embedding(timesteps, 7),
        diffusion_timestep_embedding_reference(timesteps, 7),
    )


def test_rotary_embedding_matches_reference():
    x = torch.randn(2, 4, 3, 8)
    phase = torch.randn(4, 4)
    torch.testing.assert_close(rotary_embedding(x, phase.cos(), phase.sin()), rotary_embedding_reference(x, phase.cos(), phase.sin()))

"""
Vector Add correctness tests.

Tests verify numerical accuracy across different sizes and data types.
"""
import torch
import sys
import pytest

# Add parent paths for imports
sys.path.insert(0, '.')
from kernel_ref import vector_add_reference, test_correctness


class TestVectorAddCorrectness:
    """Test suite for vector add kernel."""
    
    def test_small_size(self):
        """Test with small size (3 elements)."""
        test_correctness(size=3, dtype=torch.float32, device="cpu")
    
    def test_warp_size(self):
        """Test with warp-aligned size (32 elements)."""
        test_correctness(size=32, dtype=torch.float32, device="cpu")
    
    def test_block_size(self):
        """Test with block-aligned size (256 elements)."""
        test_correctness(size=256, dtype=torch.float32, device="cpu")
    
    def test_odd_size(self):
        """Test with odd size (not aligned to warp/block)."""
        test_correctness(size=1000007, dtype=torch.float32, device="cpu")
    
    def test_large_size(self):
        """Test with large size."""
        test_correctness(size=10000000, dtype=torch.float32, device="cpu")
    
    def test_dtype_float32(self):
        """Test with float32."""
        test_correctness(size=100000, dtype=torch.float32, device="cpu")
    
    def test_dtype_float64(self):
        """Test with float64."""
        test_correctness(size=100000, dtype=torch.float64, device="cpu")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_available(self):
        """Test that CUDA is available for GPU tests."""
        assert torch.cuda.is_available(), "CUDA not available"
    
    def test_numerical_properties(self):
        """Test numerical properties of vector add."""
        size = 1000000
        a = torch.randn(size, dtype=torch.float32)
        b = torch.randn(size, dtype=torch.float32)
        
        c = vector_add_reference(a, b)
        
        # Test commutativity
        c_swapped = vector_add_reference(b, a)
        assert torch.allclose(c, c_swapped), "Commutativity check failed"
        
        # Test identity: a + 0 = a
        zero = torch.zeros(size, dtype=torch.float32)
        c_zero = vector_add_reference(a, zero)
        assert torch.allclose(c_zero, a), "Identity check failed"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

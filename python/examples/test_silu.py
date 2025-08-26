import torch
import triton
import triton.language as tl
import pytest
import benchmark

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")


@triton.jit 
def silu_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    N,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU (Swish) activation function kernel
    
    SiLU formula: y = x * sigmoid(x) = x / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(X + offsets, mask=mask)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid_x
    tl.store(Y + offsets, y, mask=mask)


def silu(x):
    """SiLU activation function"""
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    silu_kernel[grid](
        x.view(-1), y.view(-1), N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


def test_silu():
    """Test SiLU activation against PyTorch implementation"""
    torch.manual_seed(42)
    
    # Test parameters  
    batch_size, seq_len, hidden_dim = 2, 128, 768
    
    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    
    # Triton implementation
    y_triton = silu(x)
    
    # PyTorch reference implementation  
    y_torch = torch.nn.functional.silu(x)
    
    # Compare results
    torch.testing.assert_close(y_triton, y_torch, atol=1e-5, rtol=1e-5)
    print(f"SiLU test passed! Max diff: {torch.max(torch.abs(y_triton - y_torch)):.6f}")


@pytest.mark.parametrize("batch_size, seq_len, hidden_dim", [
    (1, 64, 512),
    (2, 128, 768), 
    (4, 256, 1024),
])
def test_silu_parametrized(batch_size, seq_len, hidden_dim):
    """Parametrized test for SiLU with different sizes"""
    torch.manual_seed(42)
    
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    
    y_triton = silu(x)
    y_torch = torch.nn.functional.silu(x)
    
    torch.testing.assert_close(y_triton, y_torch, atol=1e-5, rtol=1e-5)


@benchmark.measure()
def bench_silu(batch_size, seq_len, hidden_dim, provider):
    """Benchmark SiLU activation"""
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    
    if provider == 'triton':
        silu(x)
    elif provider == 'torch':
        torch.nn.functional.silu(x)


if __name__ == "__main__":
    # Run tests
    print("Running SiLU test...")
    test_silu()
    
    # Run benchmarks
    print("\n" + "="*50)
    print("Running benchmarks...")
    
    benchmark.select_cpu_backend()
    
    # SiLU benchmarks  
    print("\nSiLU benchmarks:")
    for batch_size, seq_len, hidden_dim in [(1, 128, 768), (2, 256, 1024)]:
        for provider in ['torch', 'triton']:
            bench_silu(batch_size, seq_len, hidden_dim, provider)
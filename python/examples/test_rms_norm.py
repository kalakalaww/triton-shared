import torch
import triton
import triton.language as tl
import pytest
import benchmark

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")


@triton.jit
def rms_norm_kernel(
    X,  # pointer to input
    Y,  # pointer to output  
    W,  # pointer to weights
    stride,  # stride for rows
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """RMS Normalization kernel
    
    RMS norm formula: y = (x / rms(x)) * weight
    where rms(x) = sqrt(mean(x^2) + eps)
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    
    # Compute RMS (root mean square)
    _rms_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _rms_sq += x * x
    
    rms_sq = tl.sum(_rms_sq, axis=0) / N
    rms = tl.sqrt(rms_sq + eps)
    
    # Normalize and apply weight scaling
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x / rms) * w
        tl.store(Y + cols, y, mask=mask)


def rms_norm(x, weight, eps=1e-6):
    """RMS Normalization function"""
    input_shape = x.shape
    x = x.view(-1, x.shape[-1])
    M, N = x.shape
    
    y = torch.empty_like(x)
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("RMS norm doesn't support feature dim >= 64KB.")
    
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    
    rms_norm_kernel[(M,)](
        x, y, weight,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    
    return y.view(input_shape)


def test_rms_norm():
    """Test RMS normalization against PyTorch implementation"""
    torch.manual_seed(42)
    
    # Test parameters
    batch_size, seq_len, hidden_dim = 2, 128, 768
    eps = 1e-6
    
    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    weight = torch.ones(hidden_dim, dtype=torch.float32, device=device)
    
    # Triton implementation
    y_triton = rms_norm(x, weight, eps)
    
    # PyTorch reference implementation
    def rms_norm_torch(x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight
    
    y_torch = rms_norm_torch(x, weight, eps)
    
    # Compare results
    torch.testing.assert_close(y_triton, y_torch, atol=1e-3, rtol=1e-3)
    print(f"RMS Norm test passed! Max diff: {torch.max(torch.abs(y_triton - y_torch)):.6f}")


@pytest.mark.parametrize("batch_size, seq_len, hidden_dim", [
    (1, 64, 512),
    (2, 128, 768), 
    (4, 256, 1024),
])
def test_rms_norm_parametrized(batch_size, seq_len, hidden_dim):
    """Parametrized test for RMS norm with different sizes"""
    torch.manual_seed(42)
    eps = 1e-6
    
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    weight = torch.ones(hidden_dim, dtype=torch.float32, device=device)
    
    y_triton = rms_norm(x, weight, eps)
    
    def rms_norm_torch(x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight
    
    y_torch = rms_norm_torch(x, weight, eps)
    torch.testing.assert_close(y_triton, y_torch, atol=1e-3, rtol=1e-3)


@benchmark.measure()
def bench_rms_norm(batch_size, seq_len, hidden_dim, provider):
    """Benchmark RMS normalization"""
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    weight = torch.ones(hidden_dim, dtype=torch.float32, device=device)
    
    if provider == 'triton':
        rms_norm(x, weight)
    elif provider == 'torch':
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        x * weight


if __name__ == "__main__":
    # Run tests
    print("Running RMS Norm test...")
    test_rms_norm()
    
    # Run benchmarks
    print("\n" + "="*50)
    print("Running benchmarks...")
    
    benchmark.select_cpu_backend()
    
    # RMS Norm benchmarks
    print("\nRMS Norm benchmarks:")
    for batch_size, seq_len, hidden_dim in [(1, 128, 768), (2, 256, 1024)]:
        for provider in ['torch', 'triton']:
            bench_rms_norm(batch_size, seq_len, hidden_dim, provider)
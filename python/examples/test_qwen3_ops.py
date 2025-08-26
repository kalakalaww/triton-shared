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


@triton.jit
def gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """General Matrix Multiplication (GEMM) kernel
    
    Computes C = A @ B where A is (M, K), B is (K, N), C is (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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


def gemm(a, b):
    """General Matrix Multiplication"""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"  
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64, 
        BLOCK_SIZE_K=16,
        GROUP_SIZE_M=8
    )
    
    return c


# Test functions
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


def test_gemm():
    """Test GEMM against PyTorch implementation"""
    torch.manual_seed(42)
    
    # Test parameters
    M, K, N = 256, 512, 128
    
    # Create test data
    a = torch.randn(M, K, dtype=torch.float32, device=device)
    b = torch.randn(K, N, dtype=torch.float32, device=device)
    
    # Triton implementation
    c_triton = gemm(a, b)
    
    # PyTorch reference implementation
    c_torch = torch.matmul(a, b)
    
    # Compare results
    torch.testing.assert_close(c_triton, c_torch, atol=1e-2, rtol=1e-2)
    print(f"GEMM test passed! Max diff: {torch.max(torch.abs(c_triton - c_torch)):.6f}")


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


@pytest.mark.parametrize("M, K, N", [
    (64, 128, 64),
    (128, 256, 128),
    (256, 512, 256),
])  
def test_gemm_parametrized(M, K, N):
    """Parametrized test for GEMM with different sizes"""
    torch.manual_seed(42)
    
    a = torch.randn(M, K, dtype=torch.float32, device=device)
    b = torch.randn(K, N, dtype=torch.float32, device=device)
    
    c_triton = gemm(a, b)
    c_torch = torch.matmul(a, b)
    
    torch.testing.assert_close(c_triton, c_torch, atol=1e-2, rtol=1e-2)


# Benchmarking functions
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


@benchmark.measure()
def bench_silu(batch_size, seq_len, hidden_dim, provider):
    """Benchmark SiLU activation"""
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    
    if provider == 'triton':
        silu(x)
    elif provider == 'torch':
        torch.nn.functional.silu(x)


@benchmark.measure()
def bench_gemm(M, K, N, provider):
    """Benchmark GEMM"""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.float32, device=device)
    b = torch.randn(K, N, dtype=torch.float32, device=device)
    
    if provider == 'triton':
        gemm(a, b)
    elif provider == 'torch':
        torch.matmul(a, b)


if __name__ == "__main__":
    # Run tests
    print("Running RMS Norm test...")
    test_rms_norm()
    
    print("\nRunning SiLU test...")
    test_silu()
    
    print("\nRunning GEMM test...")  
    test_gemm()
    
    # Run benchmarks
    print("\n" + "="*50)
    print("Running benchmarks...")
    
    benchmark.select_cpu_backend()
    
    # RMS Norm benchmarks
    print("\nRMS Norm benchmarks:")
    for batch_size, seq_len, hidden_dim in [(1, 128, 768), (2, 256, 1024)]:
        for provider in ['torch', 'triton']:
            bench_rms_norm(batch_size, seq_len, hidden_dim, provider)
    
    # SiLU benchmarks  
    print("\nSiLU benchmarks:")
    for batch_size, seq_len, hidden_dim in [(1, 128, 768), (2, 256, 1024)]:
        for provider in ['torch', 'triton']:
            bench_silu(batch_size, seq_len, hidden_dim, provider)
    
    # GEMM benchmarks
    print("\nGEMM benchmarks:")
    for M, K, N in [(256, 512, 256), (512, 1024, 512)]:
        for provider in ['torch', 'triton']:
            bench_gemm(M, K, N, provider)
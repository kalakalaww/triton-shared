import torch
import triton
import triton.language as tl
import pytest
import benchmark

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")


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
    print("Running GEMM test...")  
    test_gemm()
    
    # Run benchmarks
    print("\n" + "="*50)
    print("Running benchmarks...")
    
    benchmark.select_cpu_backend()
    
    # GEMM benchmarks
    print("\nGEMM benchmarks:")
    for M, K, N in [(256, 512, 256), (512, 1024, 512)]:
        for provider in ['torch', 'triton']:
            bench_gemm(M, K, N, provider)
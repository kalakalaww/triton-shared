import torch
import triton
import triton.language as tl
import benchmark

@triton.jit
def div_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=1.0)
    c = a / b
    tl.store(c_ptr + offsets, c, mask=mask)

def div(a, b):
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    y = torch.empty_like(a)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    div_kernel[grid](
        a, b, y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def test_div():
    device = 'cpu'
    torch.manual_seed(0)
    a = torch.randn(2048, device=device)
    b = torch.randn(2048, device=device) + 1e-6  # 防止除零
    y_triton = div(a, b)
    y_torch = a / b
    assert torch.allclose(y_triton, y_torch, atol=1e-6), (y_triton, y_torch)

@benchmark.measure()
def bench_div(size, provider):
    torch.manual_seed(0)
    a = torch.randn(size, device='cpu')
    b = torch.randn(size, device='cpu') + 1e-6
    if provider == 'torch':
        a / b
    if provider == 'triton':
        div(a, b)

if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for X in [2**i for i in range(10, 14)]:
        for provider in ['torch', 'triton']:
            bench_div(X, provider)

import torch
import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver

# 如果你在用 Triton 的 CPU 支持（triton_shared），启用 driver
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")


@triton.jit
def mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    M, N,           # 当 MODE==3 (outer) 时需要用到 M,N
    MODE: tl.constexpr,     # 0: elementwise, 1: y scalar, 2: x scalar, 3: outer-product
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Mode 0: elementwise same-shape
    if MODE == 0:
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x * y
        tl.store(out_ptr + offsets, out, mask=mask)
        return

    # Mode 1: y is scalar
    if MODE == 1:
        x = tl.load(x_ptr + offsets, mask=mask)
        scalar = tl.load(y_ptr)  # single element
        out = x * scalar
        tl.store(out_ptr + offsets, out, mask=mask)
        return

    # Mode 2: x is scalar
    if MODE == 2:
        y = tl.load(y_ptr + offsets, mask=mask)
        scalar = tl.load(x_ptr)  # single element
        out = scalar * y
        tl.store(out_ptr + offsets, out, mask=mask)
        return

    # Mode 3: outer-product: rows M, cols N, produce flattened M*N elements row-major
    if MODE == 3:
        flat_idx = offsets
        # 计算 row, col（注意：这里用整除/取余，Triton 会处理向量化）
        row = flat_idx // N
        col = flat_idx - row * N
        # mask 仍然有效（超界由 mask 防护）
        a = tl.load(x_ptr + row, mask=mask)  # x is length M (shape M,1)
        b = tl.load(y_ptr + col, mask=mask)  # y is length N (shape 1,N)
        out = a * b
        tl.store(out_ptr + offsets, out, mask=mask)
        return


def mul(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    支持四种情形：
      - 完全相同 shape 的 elementwise (MODE=0)
      - 标量乘 (x is scalar 或 y is scalar) (MODE=1/2)
      - 外积广播 (x: (M,1), y: (1,N)) -> out (M,N) (MODE=3)
      - 其它复杂广播：fallback 到 torch.broadcast_to + contiguous（会分配内存）
    """
    # --- 1) 先把非张量（比如 Python float/int）转换为 tensor，避免后面访问 .dtype/.device 抛错 ---
    # 如果两者都不是张量，用默认 dtype 和 CPU 设备构造（或你可改为其它默认）
    if not torch.is_tensor(x) and not torch.is_tensor(y):
        x = torch.tensor(x, dtype=torch.get_default_dtype(), device='cpu')
        y = torch.tensor(y, dtype=x.dtype, device=x.device)
    else:
        # 如果 x 不是张量，但 y 是张量，按 y 的 dtype/device 创建 x
        if not torch.is_tensor(x) and torch.is_tensor(y):
            x = torch.tensor(x, dtype=y.dtype, device=y.device)
        # 如果 y 不是张量，但 x 是张量，按 x 的 dtype/device 创建 y
        if not torch.is_tensor(y) and torch.is_tensor(x):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)

    # --- 2) 现在可以安全地检查 dtype / device ---
    if x.dtype != y.dtype:
        raise ValueError("x 和 y 必须具有相同 dtype")
    if x.device != y.device:
        raise ValueError("x 和 y 必须在相同 device 上")

    # 以下为原有逻辑（检测 scalar / outer / elementwise 等）
    x_is_scalar = x.numel() == 1
    y_is_scalar = y.numel() == 1
    is_outer = (x.dim() == 2 and y.dim() == 2 and x.shape[1] == 1 and y.shape[0] == 1)

    if is_outer:
        # 把 (M,1) 和 (1,N) 变成一维向量以便内核按一维数组访问
        x1 = x.squeeze(1).contiguous()  # shape (M,)
        y1 = y.squeeze(0).contiguous()  # shape (N,)
        M, = x1.shape
        N, = y1.shape
        out = torch.empty((M, N), dtype=x1.dtype, device=x1.device)
        n_elements = M * N
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        mul_kernel[grid](x1, y1, out, n_elements, M, N, MODE=3, BLOCK_SIZE=BLOCK_SIZE)
        return out

    if not x_is_scalar and not y_is_scalar:
        out_shape = torch.broadcast_shapes(x.shape, y.shape)
        if x.shape == out_shape and y.shape == out_shape:
            x_c = x.contiguous()
            y_c = y.contiguous()
            out = torch.empty_like(x_c)
            n_elements = out.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul_kernel[grid](x_c, y_c, out, n_elements, 0, 0, MODE=0, BLOCK_SIZE=BLOCK_SIZE)
            return out
        else:
            x_b = torch.broadcast_to(x, out_shape).contiguous()
            y_b = torch.broadcast_to(y, out_shape).contiguous()
            out = torch.empty_like(x_b)
            n_elements = out.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul_kernel[grid](x_b, y_b, out, n_elements, 0, 0, MODE=0, BLOCK_SIZE=BLOCK_SIZE)
            return out

    if y_is_scalar:
        x_c = x.contiguous()
        scalar = y.contiguous()
        out = torch.empty_like(x_c)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        mul_kernel[grid](x_c, scalar, out, n_elements, 0, 0, MODE=1, BLOCK_SIZE=BLOCK_SIZE)
        return out

    if x_is_scalar:
        y_c = y.contiguous()
        scalar = x.contiguous()
        out = torch.empty_like(y_c)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        mul_kernel[grid](scalar, y_c, out, n_elements, 0, 0, MODE=2, BLOCK_SIZE=BLOCK_SIZE)
        return out

    raise RuntimeError("Unhandled case in mul()")

# ---------- 测试 ----------
def test_mul():
    torch.manual_seed(0)
    # 测试1：向量 * 标量
    a = torch.randn(3, device=device)
    scalar = 100.0
    out_torch = torch.mul(a, scalar)
    out_triton = mul(a, scalar)
    print("测试1 a:", a)
    print("torch.mul(a,100):", out_torch)
    print("triton mul(a,100):", out_triton)
    print("max diff:", torch.max(torch.abs(out_torch - out_triton)), "\n")

    # 测试2：outer-product (4,1) * (1,4)
    b = torch.randn(4, 1, device=device)
    c = torch.randn(1, 4, device=device)
    out_torch = torch.mul(b, c)
    out_triton = mul(b, c)
    print("测试2 b:\n", b)
    print("测试2 c:\n", c)
    print("torch.mul(b,c):\n", out_torch)
    print("triton mul(b,c):\n", out_triton)
    print("max diff:", torch.max(torch.abs(out_torch - out_triton)), "\n")

    # 测试3：elementwise same-shape
    x = torch.randn(8, device=device)
    y = torch.randn(8, device=device)
    out_torch = x * y
    out_triton = mul(x, y)
    print("测试3 x:\n", x)
    print("测试3 y:\n", y)
    print("torch.mul(x,y):\n", out_torch)
    print("triton mul(x,y):\n", out_triton)
    print("测试3 max diff:", torch.max(torch.abs(out_torch - out_triton)))


@benchmark.measure()
def bench_mul(M, N, mode, provider):
    # 根据不同模式来构造测试数据
    if mode == 'elementwise':
        # M 表示元素数量, N 在此模式下未使用
        x = torch.rand(M, device='cpu', dtype=torch.float32)
        y = torch.rand(M, device='cpu', dtype=torch.float32)
    elif mode == 'scalar':
        # M 表示元素数量, N 在此模式下未使用
        x = torch.rand(M, device='cpu', dtype=torch.float32)
        y = 3.14159  # 使用一个 Python 浮点数作为标量
    elif mode == 'outer':
        # M 和 N 分别是两个向量的维度
        x = torch.rand(M, 1, device='cpu', dtype=torch.float32)
        y = torch.rand(1, N, device='cpu', dtype=torch.float32)
    else:
        raise ValueError(f"未知的测试模式: {mode}")

    # 根据 provider 调用不同的实现
    if provider == 'torch':
       x * y  # torch 原生乘法
    elif provider == 'triton':
        mul(x, y) # 你的 triton 实现


if __name__ == "__main__":
    # 1. 首先运行正确性测试
    print("--- Correctness Tests ---")
    test_mul()
    print("\n" + "="*40 + "\n")

    # 2. 运行性能基准测试
    print("--- Performance Benchmarks ---")
    benchmark.select_cpu_backend()

    # --- 测试 Elementwise 性能 ---
    print("\n--- Benchmarking Elementwise Multiplication (M elements * M elements) ---")
    for size in [2**20, 2**22, 2**24]:
        for provider in ['torch', 'triton']:
            bench_mul(size, 0, 'elementwise', provider)

    # --- 测试 Scalar 性能 ---
    print("\n--- Benchmarking Scalar Multiplication (M elements * scalar) ---")
    for size in [2**20, 2**22, 2**24]:
        for provider in ['torch', 'triton']:
            bench_mul(size, 0, 'scalar', provider)

    # --- 测试 Outer Product 性能 ---
    # 为了让总计算量可比，我们测试 M*N, 其中 M=N=size
    print("\n--- Benchmarking Outer Product (M,1) * (1,N) ---")
    for size in [2**10, 2**11, 2**12]: # 对应总元素量为 2**20, 2**22, 2**24
        for provider in ['torch', 'triton']:
            bench_mul(size, size, 'outer', provider)




import torch
import triton
import triton.language as tl
import benchmark

# 我们可以选择性地为CPU设置后端，这在没有GPU的环境下进行测试时很有用。
# 对于实际的性能加速，通常会在GPU上运行。
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

# --- 1. Triton 内核定义 ---
@triton.jit
def sin_kernel(
    x_ptr,  # 输入张量X的指针
    output_ptr,  # 输出张量的指针
    n_elements,  # 张量中的元素总数
    BLOCK_SIZE: tl.constexpr,  # 块大小，这是一个编译时常量
):
    """
    用于计算 output = sin(x) 的 Triton 内核
    """
    # 获取当前程序实例的唯一ID
    pid = tl.program_id(axis=0)

    # 计算当前程序实例要处理的数据块的偏移量
    # tl.arange(0, BLOCK_SIZE) 生成一个 [0, 1, 2, ..., BLOCK_SIZE-1] 的数组
    # block_start 是当前块的起始索引
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建一个掩码（mask）来处理最后一个可能不满的块
    # 这一步至关重要，可以防止内存访问越界
    mask = offsets < n_elements

    # 从输入指针 x_ptr 加载数据
    # `mask=mask` 确保我们只加载在张量范围内的有效数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 对加载的数据块应用 sin 函数
    # Triton 提供了自己的数学函数，例如 tl.sin
    output = tl.sin(x)

    # 将计算结果写回到输出指针 output_ptr
    # 同样使用掩码来确保只写入有效的位置
    tl.store(output_ptr + offsets, output, mask=mask)


# --- 2. Python 封装函数 ---
def sin(x):
    """
    一个封装函数，用于启动 sin_kernel
    """
    # 为输出张量分配内存空间，形状、设备和数据类型都与输入张量相同
    output = torch.empty_like(x)

    # 检查输入张量是否是连续的，这是Triton高效工作的要求
    assert x.is_contiguous(), "Input tensor must be contiguous"

    n_elements = x.numel()

    # 定义启动内核的网格（Grid）
    # 网格决定了要启动多少个内核实例
    # triton.cdiv(a, b) 是一个天花板除法，确保所有元素都被覆盖
    # 我们将启动 n_elements / BLOCK_SIZE 个程序实例
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 启动内核
    # 我们在这里为 BLOCK_SIZE 选择一个默认值，例如 1024
    # 这个值可以被调整以获得最佳性能
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output

# --- 3. 测试与验证 ---
def test_sin(device='cpu'):
    """
    测试函数，验证我们实现的 sin 是否与 torch.sin 结果一致
    """
    torch.manual_seed(0)
    # 创建一个随机的输入张量
    size = 98432  # 选择一个不完全是块大小倍数的数字来测试掩码
    a = torch.randn(size, device=device, dtype=torch.float32)

    # 分别用我们实现的 triton_sin 和 torch.sin 进行计算
    triton_output = sin(a)
    torch_output = torch.sin(a)

    # 打印部分结果进行目视检查
    print(f"Input: {a[:5]}")
    print(f"Triton Output: {triton_output[:5]}")
    print(f"PyTorch Output: {torch_output[:5]}")

    # 使用 torch.testing.assert_close 来验证结果是否足够接近
    # atol 和 rtol 是绝对和相对容差
    torch.testing.assert_close(triton_output, torch_output, atol=1e-5, rtol=0)
    print("\n✅ Test Passed!")


# --- 4. 性能基准测试 (可选) ---
@benchmark.measure()
def bench_sin(size, provider, device='cpu'):
    """
    性能基准测试函数
    """
    a = torch.randn(size, device=device, dtype=torch.float32)
    if provider == 'torch':
        torch.sin(a)
    elif provider == 'triton':
        sin(a)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # 运行测试函数进行验证
    # 如果您有可用的 CUDA GPU，请将设备更改为 'cuda'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    test_sin(device=device)

    # 运行基准测试来比较性能
    # 仅在有GPU时运行基准测试，因为在CPU上Triton通常不如原生Torch
    if device == 'cuda':
        print("\n--- Running Benchmark ---")
        benchmark.select_backend('triton') # 使用triton的计时工具
        for size in [1024*128, 1024*1024, 1024*1024*4]:
            print(f"\nBenchmarking for size: {size}")
            for provider in ['torch', 'triton']:
                bench_sin(size, provider, device=device)
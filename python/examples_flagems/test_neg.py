import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

# 设置CPU后端（Triton核心功能依赖）
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")

@triton.jit
def neg_kernel(
    x_ptr,          # 输入张量的内存指针
    output_ptr,     # 输出张量的内存指针
    n_elements,     # 张量总元素数
    BLOCK_SIZE: tl.constexpr,  # 每个程序实例处理的元素数（编译时常量）
):
    # 获取当前程序实例ID（1D网格）
    pid = tl.program_id(axis=0)
    # 计算当前块的起始索引
    block_start = pid * BLOCK_SIZE
    # 生成当前块内所有元素的索引（偏移量）
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码：过滤掉超出总元素数的索引（防止越界）
    mask = offsets < n_elements
    # 加载输入元素（带掩码，只加载有效元素）
    x = tl.load(x_ptr + offsets, mask=mask)
    # 核心操作：取反（等价于 -x）
    output = -x
    # 存储结果（带掩码，只存储有效元素）
    tl.store(output_ptr + offsets, output, mask=mask)


def neg(x: torch.Tensor):
    """Triton实现的取反操作，对应torch.neg"""
    # 预分配输出张量（与输入同形状、同设备、同 dtype）
    output = torch.empty_like(x)
    # 计算总元素数
    n_elements = output.numel()
    # 定义并行网格：总块数 = 总元素数 ÷ 块大小（向上取整）
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 启动Triton内核（块大小设为64，适合CPU并行）
    neg_kernel[grid](x, output, n_elements, BLOCK_SIZE=64)
    return output


def test_neg():
    """测试Triton实现的neg与PyTorch原生neg的一致性"""
    # 生成测试输入（5个随机元素，与示例一致）
    torch.manual_seed(42)  # 固定随机种子，确保结果可复现
    a = torch.randn(5, device=device)
    print(f"输入张量: {a}")
    
    # 计算PyTorch原生结果（预期结果）
    expected = torch.neg(a)
    # 计算Triton实现的结果
    actual = neg(a)
    
    # 打印对比结果
    print(f"PyTorch neg结果: {expected}")
    print(f"Triton neg结果: {actual}")
    
    # 验证精度（最大差异应小于1e-6，满足浮点计算误差范围）
    max_diff = torch.max(torch.abs(expected - actual))
    print(f"最大差异: {max_diff}")
    assert max_diff < 1e-6, "结果不一致，超出误差范围"
    print("测试通过：Triton实现与PyTorch结果一致")


if __name__ == "__main__":
    test_neg()
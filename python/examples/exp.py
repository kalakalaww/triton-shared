import torch
import triton
import triton.language as tl
import math
from triton.backends.triton_shared.driver import CPUDriver

# 设置使用CPU后端
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")

@triton.jit
def exp_kernel(
    x_ptr,  # 输入向量的指针
    output_ptr,  # 输出向量的指针
    n_elements,  # 向量的大小
    BLOCK_SIZE: tl.constexpr,  # 每个程序处理的元素数量
):
    # 获取当前程序实例的ID
    pid = tl.program_id(axis=0)
    # 计算当前块的起始位置
    block_start = pid * BLOCK_SIZE
    # 计算当前块内所有元素的索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码防止越界访问
    mask = offsets < n_elements
    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask)
    # 计算指数值，使用triton.language.exp
    output = tl.exp(x)
    # 存储计算结果
    tl.store(output_ptr + offsets, output, mask=mask)


def exp(x: torch.Tensor):
    # 预分配输出张量
    output = torch.empty_like(x)
    n_elements = output.numel()
    # 定义网格调度策略
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 启动Triton内核
    exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def test_exp(device):
    # 创建测试输入: [0, ln(2)]
    x = torch.tensor([0.0, math.log(2.0)], device=device)
    
    # 使用PyTorch原生exp计算预期结果
    expected = torch.exp(x)
    
    # 使用我们实现的Triton版exp计算结果
    actual = exp(x)
    
    # 打印结果进行对比
    print(f"输入张量: {x}")
    print(f"PyTorch exp结果: {expected}")
    print(f"Triton exp结果: {actual}")
    
    # 计算并打印最大差异
    max_diff = torch.max(torch.abs(expected - actual))
    print(f"最大差异: {max_diff}")
    
    # 验证结果是否在可接受范围内
    assert max_diff < 1e-6, f"结果差异过大: {max_diff}"
    print("测试通过!")


if __name__ == "__main__":
    test_exp(device)

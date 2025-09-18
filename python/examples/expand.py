import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

# 设置Triton后端为CPU以进行测试
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")

@triton.jit
def expand_kernel(
    input_ptr,
    output_ptr,
    # 输入和输出的形状及步长信息
    # 我们将多维信息扁平化传入
    output_shape_0, output_shape_1, output_shape_2, # 示例：最多支持3维
    input_stride_0, input_stride_1, input_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    N_ELEMENTS: tl.constexpr # 输出张量的总元素个数
):
    """
    Triton kernel for the expand operation.
    每个 program 实例处理 BLOCK_SIZE 个元素。
    """
    # 1. 计算当前 program 实例要处理的元素块的起始位置
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # 创建一个大小为 BLOCK_SIZE 的偏移量数组
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 2. 创建输出掩码，防止写入越界内存
    # 当总元素数不是 BLOCK_SIZE 的整数倍时，最后一个块需要掩码
    output_mask = offsets < N_ELEMENTS

    # 3. 计算输出指针
    # 根据偏移量和输出步长计算每个元素的物理内存地址
    # 注意：这里我们处理的是扁平化后的一维索引
    output_ptrs = output_ptr + offsets

    # 4. 核心逻辑：根据输出的扁平化索引，计算输入的索引
    # 我们需要从一维的 `offsets` 恢复出多维的坐标
    # 假设我们处理一个三维张量
    # d2 = offset % shape_2
    # d1 = (offset / shape_2) % shape_1
    # d0 = (offset / shape_2) / shape_1
    d2 = offsets % output_shape_2
    d1 = (offsets // output_shape_2) % output_shape_1
    d0 = (offsets // (output_shape_1 * output_shape_2))

    # 根据 expand 的规则，如果输入维度大小为1，则其坐标应始终为0
    # 这可以通过与一个特殊构造的 "stride_mask" 相乘来实现
    # 如果 input_stride_x 是0，说明该维度大小为1，d_x * 0 = 0
    input_offsets = d0 * input_stride_0 + d1 * input_stride_1 + d2 * input_stride_2
    input_ptrs = input_ptr + input_offsets

    # 5. 从输入指针加载数据
    # 由于输入可能会有多个输出元素对应，加载时不需要掩码
    # Triton 会自动处理广播
    input_vals = tl.load(input_ptrs, mask=output_mask, other=0.0)

    # 6. 将加载的数据写入输出指针
    tl.store(output_ptrs, input_vals, mask=output_mask)


def expand(input_tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Expands a tensor to a new shape using a Triton kernel.
    
    Args:
        input_tensor: The tensor to be expanded.
        target_shape: The desired shape for the output tensor.

    Returns:
        The expanded tensor.
    """
    # 1. 检查输入合法性
    assert input_tensor.is_contiguous(), "Input tensor must be contiguous"
    
    # 2. 准备输出张量
    output_tensor = torch.empty(target_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    n_elements = output_tensor.numel()

    # 3. 处理维度对齐
    # 为了让 kernel 通用，我们将输入和输出的维度补齐到相同长度（例如3维）
    # Triton kernel 中不能有动态数量的参数
    rank = len(target_shape)
    if rank > 3:
        raise ValueError("This implementation currently supports up to 3 dimensions.")

    # 补齐形状和步长
    padded_input_shape = [1] * (3 - input_tensor.dim()) + list(input_tensor.shape)
    padded_output_shape = [1] * (3 - rank) + list(target_shape)
    
    padded_input_strides = [0] * (3 - input_tensor.dim()) + list(input_tensor.stride())
    padded_output_strides = [0] * (3 - rank) + list(output_tensor.stride())

    # expand 的核心规则：如果输入某维大小为1，而输出对应维>1，则其步长视为0
    for i in range(3):
        if padded_input_shape[i] == 1 and padded_output_shape[i] > 1:
            padded_input_strides[i] = 0

    # 4. 定义启动网格（Grid）
    # 每个 program 实例处理 BLOCK_SIZE 个元素
    BLOCK_SIZE = 1024 # 可以调整的超参数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 5. 调用 kernel
    expand_kernel[grid](
        input_tensor,
        output_tensor,
        # 传入补齐后的形状和步长
        padded_output_shape[0], padded_output_shape[1], padded_output_shape[2],
        padded_input_strides[0], padded_input_strides[1], padded_input_strides[2],
        padded_output_strides[0], padded_output_strides[1], padded_output_strides[2],
        BLOCK_SIZE=BLOCK_SIZE,
        N_ELEMENTS=n_elements
    )

    return output_tensor

def test_expand():
    """
    测试函数，验证 Triton expand 实现的正确性
    """
    torch.manual_seed(0)

    # 测试用例 1: (3, 1) -> (3, 4)
    print("--- Test Case 1: (3, 1) -> (3, 4) ---")
    a = torch.tensor([[1], [2], [3]], device=device, dtype=torch.float32)
    target_shape1 = (3, 4)
    triton_output = expand(a, target_shape1)
    torch_output = a.expand(target_shape1)
    print("Triton output:\n", triton_output)
    print("PyTorch output:\n", torch_output)
    torch.testing.assert_close(triton_output, torch_output)
    print("Test Case 1 Passed!\n")

    # 测试用例 2: (1, 3) -> (4, 3)
    print("--- Test Case 2: (1, 3) -> (4, 3) ---")
    b = torch.tensor([[10, 20, 30]], device=device, dtype=torch.float32)
    target_shape2 = (4, 3)
    triton_output2 = expand(b, target_shape2)
    torch_output2 = b.expand(target_shape2)
    print("Triton output:\n", triton_output2)
    print("PyTorch output:\n", torch_output2)
    torch.testing.assert_close(triton_output2, torch_output2)
    print("Test Case 2 Passed!\n")

    # 测试用例 3: 增加一个维度 (3, 1) -> (2, 3, 4)
    print("--- Test Case 3: (3, 1) -> (2, 3, 4) ---")
    c = torch.tensor([[1], [2], [3]], device=device, dtype=torch.float32)
    target_shape3 = (2, 3, 4)
    triton_output3 = expand(c, target_shape3)
    torch_output3 = c.expand(target_shape3)
    # print("Triton output:\n", triton_output3) # 输出较大，省略
    # print("PyTorch output:\n", torch_output3)
    torch.testing.assert_close(triton_output3, torch_output3)
    print("Test Case 3 Passed!")


if __name__ == "__main__":
    test_expand()
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver
import triton.runtime.driver
triton.runtime.driver.set_active(CPUDriver())

device = torch.device("cpu")

@triton.jit
def div_kernel(
    x_ptr, y_ptr, output_ptr, n_elements, 
    rounding_mode: tl.constexpr,  # 0: None, 1: trunc, 2: floor
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载连续张量的正确数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 移除不必要的eps（避免与PyTorch行为不一致，如需防除零可单独处理）
    out = x / y

    # 修正trunc的实现（与PyTorch保持一致：直接截断小数部分）
    if rounding_mode == 1:  # trunc
        out = tl.where(out >= 0, tl.floor(out), tl.ceil(out))
    elif rounding_mode == 2:  # floor
        out = tl.floor(out)

    tl.store(output_ptr + offsets, out, mask=mask)


def div(x, y, rounding_mode=None):
    # 处理标量输入
    if isinstance(y, (int, float)):
        y = torch.full_like(x, y, device=device)
    else:
        y = y.to(device)
    
    x = x.to(device)
    
    # 处理广播：扩展后强制转为连续张量（关键修复）
    if x.shape != y.shape:
        output_shape = torch.broadcast_shapes(x.shape, y.shape)
        # 扩展后必须用.contiguous()确保内存连续
        x = x.expand(output_shape).contiguous()
        y = y.expand(output_shape).contiguous()
    
    output = torch.empty_like(x, device=device)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    # 处理舍入模式
    rounding_code = 0
    if rounding_mode == "trunc":
        rounding_code = 1
    elif rounding_mode == "floor":
        rounding_code = 2
    elif rounding_mode is not None:
        raise ValueError("Unsupported rounding_mode: {}".format(rounding_mode))

    # 启动核函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    div_kernel[grid](x, y, output, n_elements, rounding_code, BLOCK_SIZE=BLOCK_SIZE)

    return output


def test_div():
    print("======= 标量除法 =======")
    x = torch.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637], device=device)
    scalar = 0.5

    out_triton = div(x, scalar)
    out_torch = torch.div(x, scalar)
    print("Triton 输出:", out_triton)
    print("Torch  输出:", out_torch)
    print("差异:", (out_triton - out_torch).abs().max(), "\n")

    print("======= 向量除法（广播） =======")
    a = torch.tensor([
        [-0.3711, -1.9353, -0.4605, -0.2917],
        [ 0.1815, -1.0111,  0.9805, -1.5923],
        [ 0.1062,  1.4581,  0.7759, -1.2344],
        [-0.1830, -0.0313,  1.1908, -1.4757]
    ], device=device)
    b = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308], device=device)

    out_triton = div(a, b)
    out_torch = torch.div(a, b)
    print("Triton 输出:\n", out_triton)
    print("Torch  输出:\n", out_torch)
    print("差异:\n", (out_triton - out_torch).abs().max(), "\n")

    print("======= rounding_mode='trunc' =======")
    out_triton = div(a, b, rounding_mode="trunc")
    out_torch = torch.div(a, b, rounding_mode="trunc")
    print("Triton 输出:\n", out_triton)
    print("Torch  输出:\n", out_torch)
    print("差异:\n", (out_triton - out_torch).abs().max(), "\n")

    print("======= rounding_mode='floor' =======")
    out_triton = div(a, b, rounding_mode="floor")
    out_torch = torch.div(a, b, rounding_mode="floor")
    print("Triton 输出:\n", out_triton)
    print("Torch  输出:\n", out_torch)
    print("差异:\n", (out_triton - out_torch).abs().max())


if __name__ == "__main__":
    test_div()
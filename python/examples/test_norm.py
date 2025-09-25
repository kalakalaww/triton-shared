import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
device=torch.device("cpu")


@triton.jit
def _rms_norm_fwd_fused(
    X,  # 输入指针
    Y,  # 输出指针
    W,  # 权重指针（无偏置）
    Rstd,  # 1/RMS 指针（替代原Rstd）
    stride,  # 行步长
    N,  # 输入特征数
    eps,  # 避免除零的小值
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # 计算均方根（RMS）：sqrt(mean(x^2))
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # 存储x²的部分和
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _sum_sq += x * x  # 累加x²（无需减均值）
    sum_sq = tl.sum(_sum_sq, axis=0)  # 总和
    rms = tl.sqrt(sum_sq / N + eps)  # RMS = sqrt(mean(x²) + eps)
    rstd = 1.0 / rms  # 缩放因子：1/RMS

    # 存储1/RMS（用于反向传播）
    tl.store(Rstd + row, rstd)

    # 归一化并应用权重（无偏置）
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd  # 无均值中心化，直接x / RMS
        y = x_hat * w  # 仅乘权重（无偏置）
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    DX,  # 输入梯度指针
    DY,  # 输出梯度指针
    DW,  # 权重梯度部分和指针
    X,  # 输入指针
    W,  # 权重指针
    Rstd,  # 1/RMS 指针
    Lock,  # 锁指针（用于并行归约）
    stride,  # 行步长
    N,  # 输入特征数
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    # 偏移锁和权重梯度指针
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols

    # 加载数据
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)  # 1/RMS

    # 计算输入梯度dx（因无均值，公式简化）
    x_hat = x * rstd  # 前向的x_hat = x / RMS
    wdy = w * dy
    c = tl.sum(x_hat * wdy, axis=0) / N  # 梯度中的公共项
    dx = (wdy - x_hat * c) * rstd  # RMSNorm梯度公式

    # 写入dx
    tl.store(DX + cols, dx, mask=mask)

    # 累加权重梯度dw（dw = sum(dy * x_hat)）
    partial_dw = (dy * x_hat).to(w.dtype)
    # 原子操作确保并行归约正确
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count != 0:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.atomic_xchg(Count, 1 if count == 0 else count + 1)
    tl.atomic_xchg(Lock, 0)  # 释放锁


@triton.jit
def _rms_norm_bwd_dw(
    DW,  # 权重梯度部分和指针
    FINAL_DW,  # 最终权重梯度指针
    M,  # GROUP_SIZE_M
    N,  # 输入特征数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # 归约所有部分和，得到最终dw
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])  # 展平为2D (M, N)
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)  # 存储1/RMS

        # 配置块大小（同LayerNorm逻辑）
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("RMSNorm不支持特征数 >= 64KB")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # 启动前向核函数
        _rms_norm_fwd_fused[(M,)](
            x_arg, y, weight, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        N = w.shape[0]
        # 配置并行归约参数
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # 分配中间变量
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty_like(w)
        dx = torch.empty_like(dy)

        # 展平输入
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # 启动反向核函数（计算dx和dw部分和）
        _rms_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, x, w, rstd, locks,
            x_arg.stride(0), N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps
        )

        # 归约dw部分和，得到最终dw
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _rms_norm_bwd_dw[grid](
            _dw, dw, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128
        )
        return dx, None, dw, None  # 无偏置，返回None


# 对外接口
rms_norm = RMSNorm.apply


# 测试函数（验证与PyTorch的nn.RMSNorm一致性）
def test_rms_norm(M, N, dtype, eps=1e-5, device='cpu'):
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    # 生成测试数据
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn_like(x)

    # 自定义RMSNorm前向
    y_custom = rms_norm(x, w_shape, weight, eps)
    # PyTorch原生RMSNorm前向
    torch_rms = torch.nn.RMSNorm(normalized_shape=w_shape, eps=eps, device=device, dtype=dtype)
    torch_rms.weight = torch.nn.Parameter(weight.clone())  # 共享权重
    y_torch = torch_rms(x)

    # 验证前向一致性
    assert torch.allclose(y_custom, y_torch, atol=1e-2, rtol=1e-2), "前向结果不一致"

    # 反向传播
    y_custom.backward(dy, retain_graph=True)
    dx_custom, dw_custom = x.grad.clone(), weight.grad.clone()
    x.grad, weight.grad = None, None  # 清空梯度

    y_torch.backward(dy, retain_graph=True)
    dx_torch, dw_torch = x.grad.clone(), weight.grad.clone()

    # 验证反向一致性
    assert torch.allclose(dx_custom, dx_torch, atol=1e-2, rtol=1e-2), "dx不一致"
    assert torch.allclose(dw_custom, dw_torch, atol=1e-2, rtol=1e-2), "dw不一致"
    print("测试通过！")


# 运行测试
test_rms_norm(2, 3, torch.float32)  # 对应PyTorch示例的(2,2,3)输入（展平后兼容）
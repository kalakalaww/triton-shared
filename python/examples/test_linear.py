# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
device=torch.device("cpu")


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_linear_kernel(
    # 设备张量矩阵指针
    group_a_ptrs,          # 输入矩阵指针 (类似于input)
    group_b_ptrs,          # 权重矩阵指针 (类似于weight)
    group_c_ptrs,          # 输出矩阵指针 (类似于output)
    group_bias_ptrs,       # 偏置矩阵指针 (新增)
    # 设备张量的GEMM大小，形状为[group_size, 3]
    group_gemm_sizes,      # [M, N, K]
    # 设备张量的主导维度大小，形状为[group_size, 3]
    g_lds,                 # [lda, ldb, ldc]
    # gemms数量
    group_size,
    # 是否使用偏置 (新增)
    use_bias: tl.constexpr,
    # B矩阵是否需要转置 (新增，用于实现weight^T)
    transpose_b: tl.constexpr,
    # 虚拟SM数量
    NUM_SM: tl.constexpr,
    # tile大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # 获取当前问题的gemm大小
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        
        # 计算当前GEMM问题的tile数量
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        
        # 迭代当前GEMM问题中的tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # 从当前GEMM问题选择一个tile
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            
            # 如果使用偏置，获取偏置指针
            bias_ptr = None
            if use_bias:
                bias_ptr = tl.load(group_bias_ptrs + g).to(tl.pointer_type(tl.float16))
            
            # 确定tile坐标
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles
            
            # 计算偏移量
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            
            # 根据是否需要转置B矩阵来调整指针计算
            if transpose_b:
                # B需要转置，对应 weight^T
                a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
                b_ptrs = b_ptr + offs_k[:, None] + offs_bn[None, :] * ldb
            else:
                # B不需要转置
                a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            
            # 初始化累加器
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            # 矩阵乘法计算
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # 提示编译器进行循环流水线优化
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                
                # 加载矩阵块
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                
                # 矩阵乘法
                accumulator += tl.dot(a, b)
                
                # 更新指针
                a_ptrs += BLOCK_SIZE_K
                if transpose_b:
                    b_ptrs += BLOCK_SIZE_K
                else:
                    b_ptrs += BLOCK_SIZE_K * ldb
            
            # 转换回float16
            c = accumulator.to(tl.float16)
            
            # 如果使用偏置，添加偏置
            if use_bias:
                # 加载偏置
                bias = tl.load(bias_ptr + offs_bn)
                # 广播偏置并添加到结果中
                c += bias[None, :]
            
            # 计算输出指针并存储结果
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            
            # 存储结果
            tl.store(c_ptrs, c)
            
            # 移动到下一个tile
            tile_idx += NUM_SM
        
        # 准备处理下一个gemm问题
        last_problem_end = last_problem_end + num_tiles


def group_linear_fn(group_A, group_B, group_bias=None, transpose_b=True):
    """
    实现分组的线性变换，类似于多个nn.Linear层并行计算
    output = input × weight^T + bias (如果提供了bias)
    
    参数:
        group_A: 输入矩阵列表，形状为[(M1, K1), (M2, K2), ...]
        group_B: 权重矩阵列表，形状为[(N1, K1), (N2, K2), ...] (如果transpose_b=True)
        group_bias: 偏置列表，形状为[(N1,), (N2,), ...]，可选
        transpose_b: 是否对B进行转置，默认为True，以匹配nn.Linear的weight^T行为
    """
    device = torch.device('cpu')
    assert len(group_A) == len(group_B)
    group_size = len(group_A)
    
    # 检查是否使用偏置
    use_bias = group_bias is not None
    if use_bias:
        assert len(group_A) == len(group_bias)
    
    A_addrs = []
    B_addrs = []
    C_addrs = []
    bias_addrs = [] if use_bias else None
    g_sizes = []
    g_lds = []
    group_C = []
    
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        
        # 检查维度兼容性
        if transpose_b:
            # B需要转置，所以B的形状应为(N, K)，A的形状应为(M, K)
            assert A.shape[1] == B.shape[1], f"维度不匹配: A={A.shape}, B={B.shape}, 转置后需要A[1] == B[1]"
            M, K = A.shape
            N, _ = B.shape
        else:
            # B不需要转置，所以B的形状应为(K, N)
            assert A.shape[1] == B.shape[0], f"维度不匹配: A={A.shape}, B={B.shape}, 需要A[1] == B[0]"
            M, K = A.shape
            _, N = B.shape
        
        # 创建输出矩阵
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        
        # 存储指针
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        
        if use_bias:
            bias = group_bias[i]
            assert bias.shape[0] == N, f"偏置维度不匹配: bias={bias.shape}, 应为({N},)"
            bias_addrs.append(bias.data_ptr())
        
        # 存储大小和主导维度
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
    
    # 准备设备张量
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    
    # 处理偏置
    d_bias_ptrs = None
    if use_bias:
        d_bias_ptrs = torch.tensor(bias_addrs, device=device)
    
    # 启动内核
    grid = lambda META: (META['NUM_SM'], )
    grouped_linear_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_bias_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        use_bias,
        transpose_b,
    )
    
    return group_C


# 测试代码 - 验证与PyTorch的nn.Linear功能一致性
if __name__ == "__main__":
    # 测试单个Linear层情况
    print("测试单个Linear层...")
    in_features = 20
    out_features = 30
    batch_size = 128
    
    # 创建输入和PyTorch Linear层
    input_tensor = torch.randn(batch_size, in_features, device="cpu", dtype=torch.float16)
    linear_layer = torch.nn.Linear(in_features, out_features, device="cpu", dtype=torch.float16)
    
    # 使用PyTorch计算结果
    torch_output = linear_layer(input_tensor)
    
    # 使用我们的Triton实现计算结果
    # 注意：nn.Linear的权重形状是(out_features, in_features)，我们需要转置才能用于矩阵乘法
    triton_output = group_linear_fn(
        group_A=[input_tensor],
        group_B=[linear_layer.weight],
        group_bias=[linear_layer.bias],
        transpose_b=True  # 这会将weight转置，实现input × weight^T
    )[0]
    
    # 验证结果是否接近
    print(f"PyTorch输出形状: {torch_output.shape}")
    print(f"Triton输出形状: {triton_output.shape}")
    print(f"结果误差: {torch.max(torch.abs(torch_output - triton_output))}")
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=1e-2)
    
    # 测试分组Linear情况
    print("\n测试分组Linear情况...")
    group_sizes = [(128, 20, 30), (64, 15, 25), (32, 10, 15)]  # (batch_size, in_features, out_features)
    group_A = []
    group_B = []
    group_bias = []
    torch_layers = []
    torch_outputs = []
    
    # 创建多组输入和PyTorch Linear层
    for M, K, N in group_sizes:
        A = torch.randn(M, K, device="cpu", dtype=torch.float16)
        layer = torch.nn.Linear(K, N, device="cpu", dtype=torch.float16)
        torch_output = layer(A)
        
        group_A.append(A)
        group_B.append(layer.weight)
        group_bias.append(layer.bias)
        torch_layers.append(layer)
        torch_outputs.append(torch_output)
    
    # 使用我们的Triton实现计算结果
    triton_outputs = group_linear_fn(
        group_A=group_A,
        group_B=group_B,
        group_bias=group_bias,
        transpose_b=True
    )
    
    # 验证每组结果
    for i in range(len(group_sizes)):
        print(f"组 {i+1} 误差: {torch.max(torch.abs(torch_outputs[i] - triton_outputs[i]))}")
        assert torch.allclose(torch_outputs[i], triton_outputs[i], atol=1e-2, rtol=1e-2)
    
    print("\n所有测试通过!")


# 性能基准测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作x轴的参数
        x_vals=[2**i for i in range(7, 11)],  # N的取值
        line_arg='provider',
        line_vals=['pytorch', 'triton'],  # 比较的实现
        line_names=["PyTorch nn.Linear", "Triton Linear"],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="运行时间(ms)",
        plot_name="linear-performance",
        args={},
    ))
def benchmark(N, provider):
    group_size = 4
    in_features = N
    out_features = N
    
    group_A = []
    group_B = []
    group_bias = []
    torch_layers = []
    
    for _ in range(group_size):
        # 创建输入矩阵 (batch_size, in_features)
        A = torch.randn(N, in_features, device="cpu", dtype=torch.float16)
        # 创建PyTorch Linear层
        layer = torch.nn.Linear(in_features, out_features, device="cpu", dtype=torch.float16)
        
        group_A.append(A)
        group_B.append(layer.weight)
        group_bias.append(layer.bias)
        torch_layers.append(layer)
    
    # 准备设备指针用于Triton
    A_addrs = [a.data_ptr() for a in group_A]
    B_addrs = [b.data_ptr() for b in group_B]
    C_addrs = [torch.empty_like(a[:, :out_features]).data_ptr() for a in group_A]
    bias_addrs = [b.data_ptr() for b in group_bias]
    
    g_sizes = []
    g_lds = []
    for i in range(group_size):
        M, K = group_A[i].shape
        N_out = out_features
        g_sizes += [M, N_out, K]
        g_lds += [group_A[i].stride(0), group_B[i].stride(0), N_out]
    
    d_a_ptrs = torch.tensor(A_addrs, device="cpu")
    d_b_ptrs = torch.tensor(B_addrs, device="cpu")
    d_c_ptrs = torch.tensor(C_addrs, device="cpu")
    d_bias_ptrs = torch.tensor(bias_addrs, device="cpu")
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cpu")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cpu")
    
    # 性能测试
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        # PyTorch实现
        def torch_fn():
            for i in range(group_size):
                torch_layers[i](group_A[i])
        ms, min_ms, max_ms = triton.testing.do_bench(torch_fn, quantiles=quantiles)
    else:
        # Triton实现
        def triton_fn():
            grid = lambda META: (META['NUM_SM'], )
            grouped_linear_kernel[grid](
                d_a_ptrs,
                d_b_ptrs,
                d_c_ptrs,
                d_bias_ptrs,
                d_g_sizes,
                d_g_lds,
                group_size,
                use_bias=True,
                transpose_b=True,
            )
        ms, min_ms, max_ms = triton.testing.do_bench(triton_fn, quantiles=quantiles)
    
    return ms, max_ms, min_ms


# 运行性能测试 (取消注释以运行)
# print("\n运行性能基准测试...")
# benchmark.run(show_plots=True, print_data=True)

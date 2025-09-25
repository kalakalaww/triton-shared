# Triton 实现的 Embedding Lookup（forward）和 Embedding Grad (scatter-add backward)
# 文件名: triton_qwen3_embedding.py
# 说明:
# - 包含一个高效的 forward gather kernel 和一个使用原子加的 backward scatter-add kernel
# - 附带 PyTorch 对比测试（forward + backward）
# 依赖: triton, torch

import torch
import triton
import triton.language as tl

# -----------------------------
# Kernel 参数（可调）
# -----------------------------
BLOCK_DIM = 128  # 一个内核一次处理的 embedding 维度块大小（向量化单位）

# -----------------------------
# Forward: gather kernel
# 每个 program 处理一个 (batch_pos, dim_block)
# -----------------------------
@triton.jit
def emb_fwd_kernel(
    W_ptr,          # pointer to embedding matrix W (vocab_size, hidden_dim)
    ids_ptr,        # pointer to ids flattened (batch * seq_len)
    out_ptr,        # pointer to output (batch * seq_len, hidden_dim)
    vocab_size,     # int
    hidden_dim,     # int
    num_positions,  # batch * seq_len
    stride_w_row,   # stride between rows in W (should be hidden_dim)
    stride_w_col,   # stride between cols in W (1)
    stride_out_row, # stride between rows in out (hidden_dim)
    stride_out_col, # stride between cols in out (1)
    BLOCK: tl.constexpr
):
    # program ids
    pid_pos = tl.program_id(0)   # which position (0..num_positions-1)
    pid_dim = tl.program_id(1)   # which dim-block

    # position index check
    pos = pid_pos
    if pos >= num_positions:
        return

    # ids[pos]
    id_offset = ids_ptr + pos
    idx = tl.load(id_offset, mask=pos < num_positions)
    # idx may be > vocab_size-1 in invalid inputs; we assume valid ids

    # compute row start pointer for W: W + idx * stride_w_row
    row_ptr = W_ptr + idx * stride_w_row

    # dim offsets for this block
    dim_offset = pid_dim * BLOCK + tl.arange(0, BLOCK)
    mask = dim_offset < hidden_dim

    # load from W
    src_ptr = row_ptr + dim_offset * stride_w_col
    vals = tl.load(src_ptr, mask=mask, other=0.0)

    # store to out at [pos, dim_offset]
    out_ptr_pos = out_ptr + pos * stride_out_row + dim_offset * stride_out_col
    tl.store(out_ptr_pos, vals, mask=mask)


# -----------------------------
# Backward: scatter-add kernel (atomic add to dW)
# 每个 program 处理一个 (position, dim_block)，对应 dout[pos, dim_block] -> add 到 dW[ids[pos], dim_block]
# 使用 tl.atomic_add 来解决重复 id 的累加
# -----------------------------
@triton.jit
def emb_bwd_kernel(
    gradW_ptr,      # pointer to gradient dW (vocab_size, hidden_dim)
    ids_ptr,        # pointer to ids flattened (batch * seq_len)
    grad_out_ptr,   # pointer to grad wrt output (batch * seq_len, hidden_dim)
    vocab_size,
    hidden_dim,
    num_positions,
    stride_gw_row,
    stride_gw_col,
    stride_go_row,
    stride_go_col,
    BLOCK: tl.constexpr
):
    pid_pos = tl.program_id(0)
    pid_dim = tl.program_id(1)

    pos = pid_pos
    if pos >= num_positions:
        return

    # load id
    id_offset = ids_ptr + pos
    idx = tl.load(id_offset, mask=pos < num_positions)

    # dim offsets
    dim_offset = pid_dim * BLOCK + tl.arange(0, BLOCK)
    mask = dim_offset < hidden_dim

    # load grad_out
    grad_out_ptr_pos = grad_out_ptr + pos * stride_go_row + dim_offset * stride_go_col
    vals = tl.load(grad_out_ptr_pos, mask=mask, other=0.0)

    # target pointer in gradW: gradW[idx, dim_offset]
    gw_ptr = gradW_ptr + idx * stride_gw_row + dim_offset * stride_gw_col

    # atomic add
    tl.atomic_add(gw_ptr, vals, mask=mask)


# -----------------------------
# Python wrappers
# -----------------------------
def embedding_forward_triton(W: torch.Tensor, ids: torch.LongTensor):
    """W: (vocab_size, hidden_dim), ids: (batch, seq_len)
    returns: out (batch, seq_len, hidden_dim)
    """
    assert W.is_contiguous()
    batch, seq_len = ids.shape
    num_positions = batch * seq_len
    vocab_size, hidden_dim = W.shape

    ids_flat = ids.reshape(-1).contiguous()

    out = torch.empty((num_positions, hidden_dim), device=W.device, dtype=W.dtype)

    grid = (num_positions, (hidden_dim + BLOCK_DIM - 1) // BLOCK_DIM)

    emb_fwd_kernel[grid](
        W,                                  # W_ptr auto-converted
        ids_flat,                           # ids_ptr
        out,                                 # out_ptr
        vocab_size,
        hidden_dim,
        num_positions,
        hidden_dim,  # stride_w_row
        1,           # stride_w_col
        hidden_dim,  # stride_out_row
        1,           # stride_out_col
        BLOCK=BLOCK_DIM
    )

    return out.view(batch, seq_len, hidden_dim)


def embedding_backward_triton(vocab_size: int, hidden_dim: int, ids: torch.LongTensor, grad_out: torch.Tensor):
    """Compute gradient wrt W given ids and grad_out (shape batch, seq_len, hidden_dim)
    returns gradW: (vocab_size, hidden_dim)
    """
    batch, seq_len, hd = grad_out.shape
    assert hd == hidden_dim
    num_positions = batch * seq_len

    ids_flat = ids.reshape(-1).contiguous()
    grad_out_flat = grad_out.reshape(num_positions, hidden_dim).contiguous()

    gradW = torch.zeros((vocab_size, hidden_dim), device=grad_out.device, dtype=grad_out.dtype)

    grid = (num_positions, (hidden_dim + BLOCK_DIM - 1) // BLOCK_DIM)

    emb_bwd_kernel[grid](
        gradW,
        ids_flat,
        grad_out_flat,
        vocab_size,
        hidden_dim,
        num_positions,
        hidden_dim,  # stride_gw_row
        1,           # stride_gw_col
        hidden_dim,  # stride_go_row
        1,           # stride_go_col
        BLOCK=BLOCK_DIM
    )

    return gradW


# -----------------------------
# 测试脚本: 与 PyTorch 对比
# -----------------------------
if __name__ == '__main__':
    import time

    device = 'cuda'
    dtype = torch.float32

    # 小规模示例（本地 run 可调）
    vocab_size = 20000
    hidden_dim = 256
    batch = 4
    seq_len = 16

    torch.manual_seed(0)

    W = torch.randn(vocab_size, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    ids = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)

    # --- forward compare ---
    t0 = time.time()
    out_triton = embedding_forward_triton(W.detach(), ids)
    t1 = time.time()
    out_torch = torch.nn.functional.embedding(ids, W)
    t2 = time.time()

    print(f"Forward: triton time {t1-t0:.6f}s, torch time {t2-t1:.6f}s")
    diff = (out_triton - out_torch).abs().max().item()
    print(f"Forward max abs diff: {diff:e}")

    # --- backward compare ---
    dout = torch.randn_like(out_torch)

    # PyTorch grad
    W.grad = None
    out_torch.backward(dout)
    grad_torch = W.grad.clone()

    # Triton grad
    grad_triton = embedding_backward_triton(vocab_size, hidden_dim, ids, dout)

    diffg = (grad_triton - grad_torch).abs().max().item()
    print(f"Backward max abs diff: {diffg:e}")

    # Basic correctness assertions (tolerance depends on fp32 atomic ordering)
    assert diff < 1e-5, f"forward mismatch {diff}"
    assert diffg < 1e-4, f"backward mismatch {diffg}"

    print('All tests passed.')

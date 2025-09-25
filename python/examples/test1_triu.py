# triu_operation.py
import torch
import triton
import triton.language as tl

# --- Triton Kernel for Triu ---
@triton.jit
def triu_kernel(
    input_ptr,
    output_ptr,
    M, N,
    input_stride_m,
    input_stride_n,
    diagonal,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel to compute the upper triangular part of a matrix.
    """
    # Each program instance handles one row
    row_idx = tl.program_id(0)

    # Iterate over columns in blocks
    for col_block_start in range(0, N, BLOCK_SIZE_N):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < N

        # Create the upper-triangular mask
        # An element is in the upper triangle if its column index is >= row index + diagonal
        triu_mask = col_offsets >= (row_idx + diagonal)

        # Calculate memory offsets for input and output (they are the same)
        offsets = row_idx * input_stride_m + col_offsets * input_stride_n
        
        # Load values from the input tensor
        # 'other=0.0' ensures out-of-bounds loads are treated as zero
        values = tl.load(input_ptr + offsets, mask=col_mask, other=0.0)

        # Apply the triu logic: keep values where triu_mask is true, otherwise set to 0.0
        output_values = tl.where(triu_mask, values, 0.0)
        
        # Store the result to the output tensor
        tl.store(output_ptr + offsets, output_values, mask=col_mask)

# --- Python Wrapper for Triu ---
def triu(input_tensor, diagonal=0):
    """
    Returns the upper triangular part of a 2D tensor.
    """
    assert input_tensor.ndim == 2, "triu operation only supports 2D tensors."
    M, N = input_tensor.shape
    output = torch.empty_like(input_tensor)

    # Grid is 1D, with one program per row
    grid = (M,)
    BLOCK_SIZE_N = 1024 # A reasonably large block size for columns

    triu_kernel[grid](
        input_tensor,
        output,
        M, N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        diagonal,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output

# --- Test Function for Triu ---
def test_triu():
    """Tests the triu (upper triangular) operation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    x = torch.randn(5, 5, dtype=torch.float32, device=device)

    # Test with default diagonal
    y_triton = triu(x)
    y_torch = torch.triu(x)
    torch.testing.assert_close(y_triton, y_torch)
    print("Triu test (diagonal=0) PASSED!")

    # Test with positive diagonal
    y_triton_pos = triu(x, diagonal=1)
    y_torch_pos = torch.triu(x, diagonal=1)
    torch.testing.assert_close(y_triton_pos, y_torch_pos)
    print("Triu test (diagonal=1) PASSED!")

    # Test with negative diagonal
    y_triton_neg = triu(x, diagonal=-1)
    y_torch_neg = torch.triu(x, diagonal=-1)
    torch.testing.assert_close(y_triton_neg, y_torch_neg)
    print("Triu test (diagonal=-1) PASSED!")


if __name__ == "__main__":
    test_triu()
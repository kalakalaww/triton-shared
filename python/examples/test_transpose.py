# transpose_operation.py
import torch
import triton
import triton.language as tl

# --- Triton Kernel for 2D Transpose ---
@triton.jit
def transpose_2d_kernel(
    input_ptr,
    output_ptr,
    M, N,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for transposing a 2D tensor (matrix).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the offsets for the block this program is responsible for
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create masks to handle tensors that are not perfectly divisible by block sizes
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # Calculate memory offsets
    input_offsets = offsets_m[:, None] * input_stride_0 + offsets_n[None, :] * input_stride_1
    output_offsets = offsets_n[:, None] * output_stride_0 + offsets_m[None, :] * output_stride_1
    
    # Load the input block and store it to the transposed output block
    input_block = tl.load(input_ptr + input_offsets, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(output_ptr + output_offsets, input_block, mask=mask_n[:, None] & mask_m[None, :])


# --- Python Wrapper for Transpose ---
def transpose(input_tensor, dim0=0, dim1=1):
    """
    Transposes two dimensions of a tensor. Uses a Triton kernel for 2D tensors.
    """
    if input_tensor.ndim == 2:
        M, N = input_tensor.shape
        output = torch.empty((N, M), dtype=input_tensor.dtype, device=input_tensor.device)
        
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        transpose_2d_kernel[grid](
            input_tensor, output,
            M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        return output
    else:
        # Fallback to PyTorch's permute for tensors with more than 2 dimensions
        perm = list(range(input_tensor.ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return input_tensor.permute(perm)

# --- Test Function for Transpose ---
def test_transpose():
    """Tests the transpose operation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test 2D transpose
    x_2d = torch.randn(30, 40, dtype=torch.float32, device=device)
    y_triton_2d = transpose(x_2d)
    y_torch_2d = torch.transpose(x_2d, 0, 1)
    torch.testing.assert_close(y_triton_2d, y_torch_2d)
    print(f"Transpose 2D test PASSED! Shape: {x_2d.shape} -> {y_triton_2d.shape}")

    # Test 4D transpose (fallback)
    x_4d = torch.randn(2, 3, 4, 5, dtype=torch.float32, device=device)
    y_triton_4d = transpose(x_4d, 1, 3)
    y_torch_4d = torch.transpose(x_4d, 1, 3)
    torch.testing.assert_close(y_triton_4d, y_torch_4d)
    print(f"Transpose 4D test (fallback) PASSED! Shape: {x_4d.shape} -> {y_triton_4d.shape}")

if __name__ == "__main__":
    test_transpose()
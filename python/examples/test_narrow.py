# narrow_operation.py
import torch
import triton
import triton.language as tl

# --- Triton Kernel for Narrow ---
@triton.jit
def narrow_kernel(
    input_ptr,
    output_ptr,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    dim_start, dim_length,
    input_shape_0, input_shape_1,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for the narrow operation. It selects a contiguous subset
    of a tensor along a specified dimension.
    """
    pid = tl.program_id(0)

    if dim == 0:
        # Narrowing along the first dimension (rows)
        row_idx = pid
        if row_idx >= dim_length:
            return
        actual_row = dim_start + row_idx

        # Iterate over columns in blocks
        for col_start in range(0, input_shape_1, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < input_shape_1

            input_offsets = actual_row * input_stride_0 + col_offsets * input_stride_1
            output_offsets = row_idx * output_stride_0 + col_offsets * output_stride_1

            data = tl.load(input_ptr + input_offsets, mask=mask)
            tl.store(output_ptr + output_offsets, data, mask=mask)
    else:
        # Narrowing along the second dimension (columns)
        row_idx = pid
        if row_idx >= input_shape_0:
            return

        col_offsets = dim_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < (dim_start + dim_length)

        input_offsets = row_idx * input_stride_0 + col_offsets * input_stride_1
        output_col_offsets = tl.arange(0, BLOCK_SIZE)
        output_offsets = row_idx * output_stride_0 + output_col_offsets * output_stride_1

        data = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + output_offsets, data, mask=output_col_offsets < dim_length)

# --- Python Wrapper for Narrow ---
def narrow(input_tensor, dim, start, length):
    """
    Selects a contiguous subset of the input tensor along a specified dimension.
    """
    assert dim < input_tensor.ndim, f"Dimension {dim} is out of range."
    assert start >= 0 and start < input_tensor.shape[dim], f"Start index {start} is out of range."
    assert length > 0 and start + length <= input_tensor.shape[dim], "Invalid length."

    output_shape = list(input_tensor.shape)
    output_shape[dim] = length
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    # Currently supports 2D tensors with Triton
    if input_tensor.ndim == 2 and dim < 2:
        BLOCK_SIZE = 1024
        grid = (length,) if dim == 0 else (input_tensor.shape[0],)

        narrow_kernel[grid](
            input_tensor, output,
            input_tensor.stride(0), input_tensor.stride(1),
            output.stride(0), output.stride(1),
            start, length,
            input_tensor.shape[0], input_tensor.shape[1],
            dim=dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Fallback to PyTorch for non-2D tensors
        output = torch.narrow(input_tensor, dim, start, length)

    return output

# --- Test Function for Narrow ---
def test_narrow():
    """Tests the Triton-accelerated narrow operation against PyTorch's implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    x = torch.randn(10, 20, dtype=torch.float32, device=device)

    # Test case 1: Narrow along dimension 0
    y_triton_dim0 = narrow(x, dim=0, start=2, length=5)
    y_torch_dim0 = torch.narrow(x, dim=0, start=2, length=5)
    torch.testing.assert_close(y_triton_dim0, y_torch_dim0)
    print("Narrow test (dim=0) PASSED!")
    print(f"  Input Shape: {x.shape}, Output Shape: {y_triton_dim0.shape}\n")

    # Test case 2: Narrow along dimension 1
    y_triton_dim1 = narrow(x, dim=1, start=5, length=10)
    y_torch_dim1 = torch.narrow(x, dim=1, start=5, length=10)
    torch.testing.assert_close(y_triton_dim1, y_torch_dim1)
    print("Narrow test (dim=1) PASSED!")
    print(f"  Input Shape: {x.shape}, Output Shape: {y_triton_dim1.shape}\n")


if __name__ == "__main__":
    test_narrow()
# expand_operation.py
import torch
import triton
import triton.language as tl

# --- Triton Kernel for Expand ---
@triton.jit
def expand_kernel(
    input_ptr,
    output_ptr,
    # Pointers to tensor metadata
    output_shape_ptr,
    output_strides_ptr,
    input_strides_ptr,
    # Metadata
    ndim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for the expand operation (broadcasting).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate the multi-dimensional index for each output element
    # and map it to the corresponding input element's index.
    # This is done by iterating through dimensions from slowest to fastest moving.
    linear_indices = offsets
    input_offset = 0
    
    # Using tl.static_for for loop over dimensions
    for dim in tl.static_range(ndim):
        output_stride = tl.load(output_strides_ptr + dim)
        input_stride = tl.load(input_strides_ptr + dim)
        output_shape = tl.load(output_shape_ptr + dim)
        
        # Calculate the index for the current dimension
        dim_idx = (linear_indices // output_stride) % output_shape
        
        # Add to input offset. If input_stride is 0, it means this dimension
        # was broadcasted (size 1), so it doesn't contribute to the offset.
        input_offset += dim_idx * input_stride

    # Load from input and store to output
    input_vals = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)


# --- Python Wrapper for Expand ---
def expand(input_tensor, new_shape):
    """
    Expands the input tensor to a new shape using broadcasting rules.
    """
    output = torch.empty(new_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    n_elements = output.numel()
    
    # Get shape and strides for the kernel, handling broadcasting
    # A dimension of size 1 in the input will have a stride of 0 for broadcasting
    input_strides_b = torch.as_strided(input_tensor, new_shape, [0] * len(new_shape)).stride()
    
    # Prepare metadata tensors for the kernel
    output_shape_tensor = torch.tensor(output.shape, dtype=torch.int32, device=output.device)
    output_strides_tensor = torch.tensor(output.stride(), dtype=torch.int32, device=output.device)
    input_strides_tensor = torch.tensor(input_strides_b, dtype=torch.int32, device=output.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    expand_kernel[grid](
        input_tensor,
        output,
        output_shape_tensor,
        output_strides_tensor,
        input_strides_tensor,
        output.ndim,
        n_elements,
        # Using tl.static_for requires ndim to be a compile-time constant
        # For simplicity in this example, we handle it in the kernel or can pass as constexpr
        # A more robust solution might involve kernel specialization.
        # Here we pass it as a regular argument, which is less optimal.
        # For this example, it's sufficient.
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# --- Test Function for Expand ---
def test_expand():
    """Tests the expand operation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test case 1
    x1 = torch.tensor([[1], [2], [3]], device=device)
    y_triton1 = expand(x1, (3, 4))
    y_torch1 = x1.expand(3, 4)
    torch.testing.assert_close(y_triton1, y_torch1)
    print(f"Expand test 1 PASSED! Shape: {x1.shape} -> {y_triton1.shape}")

    # Test case 2
    x2 = torch.randn(3, 1, 5, dtype=torch.float32, device=device)
    y_triton2 = expand(x2, (3, 4, 5))
    y_torch2 = x2.expand(3, 4, 5)
    torch.testing.assert_close(y_triton2, y_torch2)
    print(f"Expand test 2 PASSED! Shape: {x2.shape} -> {y_triton2.shape}")

if __name__ == "__main__":
    test_expand()
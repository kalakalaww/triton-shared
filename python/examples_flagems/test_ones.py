# fill_operations.py
import torch
import triton
import triton.language as tl

# --- Triton Kernel for Filling a Tensor ---
@triton.jit
def fill_kernel(
    output_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to fill a tensor with a specified scalar value.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tl.store(output_ptr + offsets, value, mask=mask)

# --- Python Wrapper Functions ---
def ones(shape, dtype=torch.float32, device=None):
    """Creates a tensor of the given shape filled with ones."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output = torch.empty(shape, dtype=dtype, device=device)
    n_elements = output.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fill_kernel[grid](output, n_elements, 1.0, BLOCK_SIZE=BLOCK_SIZE)
    return output

def zeros(shape, dtype=torch.float32, device=None):
    """Creates a tensor of the given shape filled with zeros."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    output = torch.empty(shape, dtype=dtype, device=device)
    n_elements = output.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fill_kernel[grid](output, n_elements, 0.0, BLOCK_SIZE=BLOCK_SIZE)
    return output

def ones_like(input_tensor):
    """Creates a tensor of ones with the same shape and type as the input."""
    return ones(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)

# --- Test Function for Fill Operations ---
def test_ones_zeros():
    """Tests the ones, zeros, and ones_like operations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (3, 4, 5)

    # Test ones
    y_triton_ones = ones(shape, dtype=torch.float32, device=device)
    y_torch_ones = torch.ones(shape, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton_ones, y_torch_ones)
    print(f"Ones test PASSED! Shape: {y_triton_ones.shape}")

    # Test zeros
    y_triton_zeros = zeros(shape, dtype=torch.float32, device=device)
    y_torch_zeros = torch.zeros(shape, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton_zeros, y_torch_zeros)
    print(f"Zeros test PASSED! Shape: {y_triton_zeros.shape}")

    # Test ones_like
    x = torch.randn(2, 3, dtype=torch.float32, device=device)
    y_triton_like = ones_like(x)
    y_torch_like = torch.ones_like(x)
    torch.testing.assert_close(y_triton_like, y_torch_like)
    print(f"Ones_like test PASSED! Shape: {y_triton_like.shape}")

if __name__ == "__main__":
    test_ones_zeros()
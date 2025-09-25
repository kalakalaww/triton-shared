# arange_operation.py
import torch
import triton
import triton.language as tl

# --- Triton Kernel for Arange ---
@triton.jit
def arange_kernel(
    output_ptr,
    start,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to generate a range of values.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = start + offsets * step
    tl.store(output_ptr + offsets, values, mask=mask)

# --- Python Wrapper for Arange ---
def arange(start, end=None, step=1, dtype=torch.float32, device=None):
    """
    Creates a 1-D tensor of size `floor((end - start) / step)` with values
    from the interval [start, end) taken with common difference step.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if end is None:
        end = start
        start = 0

    n_elements = int((end - start + step - 1) / step) if step > 0 else int((end - start + step + 1) / step)
    output = torch.empty(n_elements, dtype=dtype, device=device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    arange_kernel[grid](
        output,
        float(start),
        float(step),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# --- Test Function for Arange ---
def test_arange():
    """Tests the Triton-accelerated arange operation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test case 1: arange(end)
    y_triton = arange(10, dtype=torch.float32, device=device)
    y_torch = torch.arange(10, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton, y_torch)
    print("Arange test (0-10, step=1) PASSED!")

    # Test case 2: arange(start, end, step)
    y_triton_step = arange(5, 15, 2, dtype=torch.float32, device=device)
    y_torch_step = torch.arange(5, 15, 2, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton_step, y_torch_step)
    print("Arange test (5-15, step=2) PASSED!")

if __name__ == "__main__":
    test_arange()
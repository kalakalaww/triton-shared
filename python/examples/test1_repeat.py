import torch
import triton
import triton.language as tl
import sys

# --------------------------------------------------------------------------
# 1. Triton Kernel for the 'repeat' operation
#    (This part will only be callable if a GPU is present)
# --------------------------------------------------------------------------
@triton.jit
def repeat_kernel(
    input_ptr,
    output_ptr,
    output_n_elements,
    input_shape_0,
    input_shape_1,
    output_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_n_elements
    output_idx_dim0 = offsets // output_shape_1
    output_idx_dim1 = offsets % output_shape_1
    input_idx_dim0 = output_idx_dim0 % input_shape_0
    input_idx_dim1 = output_idx_dim1 % input_shape_1
    input_offsets = input_idx_dim0 * input_shape_1 + input_idx_dim1
    input_values = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + offsets, input_values, mask=mask)


# --------------------------------------------------------------------------
# 2. Python wrapper function to launch the Triton kernel
# --------------------------------------------------------------------------
def triton_repeat(x: torch.Tensor, repeats: tuple) -> torch.Tensor:
    if not x.is_cuda:
        raise TypeError("Input tensor for Triton kernel must be a CUDA tensor.")
    if x.dim() != 2 or len(repeats) != 2:
        raise ValueError("This Triton implementation currently only supports 2D tensors and a repeats tuple of length 2.")
    
    input_shape = x.shape
    output_shape = (input_shape[0] * repeats[0], input_shape[1] * repeats[1])
    output_n_elements = output_shape[0] * output_shape[1]
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(output_n_elements, meta['BLOCK_SIZE']),)

    repeat_kernel[grid](
        x,
        output,
        output_n_elements,
        input_shape[0],
        input_shape[1],
        output_shape[1],
        BLOCK_SIZE=1024,
    )
    return output


# --------------------------------------------------------------------------
# 3. Main execution block for comparison testing
# --------------------------------------------------------------------------
if __name__ == "__main__":

    # 检查是否有可用的CUDA设备
    if torch.cuda.is_available():
        print("✅ NVIDIA GPU found. Running full comparison on 'cuda' device.\n")
        device = 'cuda'
    else:
        print("⚠️ WARNING: No NVIDIA GPU found. Triton part will be skipped.")
        print("           Running torch-only part on 'cpu' device.\n")
        device = 'cpu'

    # --- Test Case 1: Random Floating-Point Tensor ---
    print("--- Test Case 1: Random Floating-Point Tensor ---")
    
    try:
        input_tensor_1 = torch.randn((64, 128), device=device, dtype=torch.float32)
        repeats_1 = (2, 3)

        # PyTorch part can run on both CPU and GPU
        torch_result_1 = input_tensor_1.repeat(repeats_1)
        
        print(f"Input Shape: {input_tensor_1.shape}")
        print(f"Repeats: {repeats_1}")
        print(f"PyTorch Result Shape: {torch_result_1.shape}")

        # Triton part only runs if GPU is available
        if device == 'cuda':
            triton_result_1 = triton_repeat(input_tensor_1, repeats_1)
            print(f"Triton Result Shape: {triton_result_1.shape}")
            are_equal_1 = torch.allclose(torch_result_1, triton_result_1, atol=1e-5, rtol=1e-5)
            if are_equal_1:
                print("\n✅ Test Passed: The results of Triton and PyTorch are numerically identical.")
            else:
                print("\n❌ Test Failed: The results are different.")
        else:
            print("\n🔵 Triton test skipped (requires NVIDIA GPU).")

    except Exception as e:
        print(f"\nAn error occurred during Test Case 1: {e}")


    print("\n" + "="*50 + "\n")

    # --- Test Case 2: Small Integer Tensor for Visual Inspection ---
    print("--- Test Case 2: Small Integer Tensor for Visual Inspection ---")
    
    try:
        input_tensor_2 = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.float32)
        repeats_2 = (3, 2)
        
        torch_result_2 = input_tensor_2.repeat(repeats_2)

        print(f"Input Tensor:\n{input_tensor_2.cpu().numpy()}\n")
        print(f"Repeats: {repeats_2}\n")
        print(f"PyTorch Repeat Result:\n{torch_result_2.cpu().numpy()}\n")

        if device == 'cuda':
            triton_result_2 = triton_repeat(input_tensor_2, repeats_2)
            print(f"Triton Repeat Result:\n{triton_result_2.cpu().numpy()}\n")
            are_equal_2 = torch.equal(torch_result_2, triton_result_2)
            if are_equal_2:
                print("✅ Test Passed: The results for the second case are exactly identical.")
            else:
                print("❌ Test Failed: The results for the second case are different.")
        else:
             print("🔵 Triton test skipped (requires NVIDIA GPU).")

    except Exception as e:
        print(f"\nAn error occurred during Test Case 2: {e}")
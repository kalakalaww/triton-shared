# view_operations.py
import torch

def unsqueeze(input_tensor, dim):
    """
    Adds a dimension of size 1 at the specified position.
    This is a metadata-only operation and wraps torch.unsqueeze.
    """
    return torch.unsqueeze(input_tensor, dim)

def reshape(input_tensor, shape):
    """
    Reshapes a tensor to a new shape.
    This is a metadata-only operation if the number of elements is unchanged
    and the tensor is contiguous. It wraps torch.reshape.
    """
    return input_tensor.reshape(shape)

# --- Test Functions ---
def test_unsqueeze():
    """Tests the unsqueeze operation."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)

    y_wrapper = unsqueeze(x, 0)
    y_torch = torch.unsqueeze(x, 0)
    torch.testing.assert_close(y_wrapper, y_torch)
    print(f"Unsqueeze test (dim=0) PASSED! Shape: {x.shape} -> {y_wrapper.shape}")

    y_wrapper = unsqueeze(x, -1)
    y_torch = torch.unsqueeze(x, -1)
    torch.testing.assert_close(y_wrapper, y_torch)
    print(f"Unsqueeze test (dim=-1) PASSED! Shape: {x.shape} -> {y_wrapper.shape}")

def test_reshape():
    """Tests the reshape operation."""
    torch.manual_seed(42)
    x = torch.randn(6, 4)

    y_wrapper = reshape(x, (3, 8))
    y_torch = x.reshape(3, 8)
    torch.testing.assert_close(y_wrapper, y_torch)
    print(f"Reshape test PASSED! Shape: {x.shape} -> {y_wrapper.shape}")
    
    y_wrapper = reshape(x, (-1, 12))
    y_torch = x.reshape(-1, 12)
    torch.testing.assert_close(y_wrapper, y_torch)
    print(f"Reshape with -1 test PASSED! Shape: {x.shape} -> {y_wrapper.shape}")

if __name__ == "__main__":
    print("--- Testing Unsqueeze ---")
    test_unsqueeze()
    print("\n--- Testing Reshape ---")
    test_reshape()
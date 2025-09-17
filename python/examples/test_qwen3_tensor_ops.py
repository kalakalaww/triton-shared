import torch
import triton
import triton.language as tl
import pytest

from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")


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
    """Narrow operation kernel - select a contiguous subset along a dimension"""
    pid = tl.program_id(0)
    
    if dim == 0:
        row_idx = pid
        if row_idx >= dim_length:
            return
        actual_row = dim_start + row_idx
        
        for col_start in range(0, input_shape_1, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < input_shape_1
            
            input_offsets = actual_row * input_stride_0 + col_offsets * input_stride_1
            output_offsets = row_idx * output_stride_0 + col_offsets * output_stride_1
            
            data = tl.load(input_ptr + input_offsets, mask=mask)
            tl.store(output_ptr + output_offsets, data, mask=mask)
    else:
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


@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill tensor with a constant value (ones, zeros, or any scalar)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def arange_kernel(
    output_ptr,
    start,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate a range of values"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    values = start + offsets * step
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def repeat_kernel(
    input_ptr,
    output_ptr,
    input_shape_ptr,
    output_shape_ptr,
    input_strides_ptr,
    output_strides_ptr,
    ndim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Repeat tensor elements along specified dimensions"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    for idx in range(BLOCK_SIZE):
        linear_idx = block_start + idx
        if linear_idx >= n_elements:
            return
            
        output_idx = linear_idx
        input_offset = 0
        
        for dim in range(ndim):
            output_stride = tl.load(output_strides_ptr + dim)
            input_stride = tl.load(input_strides_ptr + dim)
            output_size = tl.load(output_shape_ptr + dim)
            input_size = tl.load(input_shape_ptr + dim)
            
            if output_size > 0:
                dim_idx = (output_idx // output_stride) % output_size
                input_dim_idx = dim_idx % input_size
                input_offset += input_dim_idx * input_stride
        
        value = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + linear_idx, value)


@triton.jit
def transpose_2d_kernel(
    input_ptr,
    output_ptr,
    M, N,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """Transpose a 2D tensor"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    for m_idx in range(BLOCK_SIZE):
        if m_offsets[m_idx] >= M:
            continue
        for n_idx in range(BLOCK_SIZE):
            if n_offsets[n_idx] >= N:
                continue
                
            input_offset = m_offsets[m_idx] * input_stride_0 + n_offsets[n_idx] * input_stride_1
            output_offset = n_offsets[n_idx] * output_stride_0 + m_offsets[m_idx] * output_stride_1
            
            value = tl.load(input_ptr + input_offset)
            tl.store(output_ptr + output_offset, value)


@triton.jit
def expand_kernel(
    input_ptr,
    output_ptr,
    input_shape_ptr,
    output_shape_ptr,
    input_strides_ptr,
    output_strides_ptr,
    ndim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Expand tensor to a new shape by broadcasting"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    for idx in range(BLOCK_SIZE):
        linear_idx = block_start + idx
        if linear_idx >= n_elements:
            return
            
        output_idx = linear_idx
        input_offset = 0
        
        for dim in range(ndim):
            output_stride = tl.load(output_strides_ptr + dim)
            input_stride = tl.load(input_strides_ptr + dim)
            output_shape = tl.load(output_shape_ptr + dim)
            input_shape = tl.load(input_shape_ptr + dim)
            
            dim_idx = (output_idx // output_stride) % output_shape
            
            if input_shape == 1:
                input_dim_idx = 0
            else:
                input_dim_idx = dim_idx
                
            input_offset += input_dim_idx * input_stride
            
        value = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + linear_idx, value)


@triton.jit
def triu_kernel(
    input_ptr,
    output_ptr,
    M, N,
    diagonal,
    BLOCK_SIZE: tl.constexpr,
):
    """Upper triangular part of a matrix"""
    pid = tl.program_id(0)
    
    row = pid // ((N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    col_block = pid % ((N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    if row >= M:
        return
        
    col_start = col_block * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    triu_mask = cols >= (row + diagonal)
    
    input_offsets = row * N + cols
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    output_values = tl.where(triu_mask, values, 0.0)
    tl.store(output_ptr + input_offsets, output_values, mask=mask)


def narrow(input_tensor, dim, start, length):
    """Narrow operation - select a contiguous subset along a dimension"""
    assert dim < input_tensor.ndim, f"dim {dim} out of range for tensor with {input_tensor.ndim} dimensions"
    assert start >= 0 and start < input_tensor.shape[dim], f"start {start} out of range for dimension {dim} with size {input_tensor.shape[dim]}"
    assert length > 0 and start + length <= input_tensor.shape[dim], f"Invalid length {length} for start {start} and dimension size {input_tensor.shape[dim]}"
    
    output_shape = list(input_tensor.shape)
    output_shape[dim] = length
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    if input_tensor.ndim == 2 and dim < 2:
        BLOCK_SIZE = 1024
        if dim == 0:
            grid = (length,)
        else:
            grid = (input_tensor.shape[0],)
            
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
        input_flat = input_tensor.flatten()
        output_flat = output.flatten()
        
        total_before = 1
        for i in range(dim):
            total_before *= input_tensor.shape[i]
        dim_size = input_tensor.shape[dim]
        total_after = 1
        for i in range(dim + 1, input_tensor.ndim):
            total_after *= input_tensor.shape[i]
            
        for i in range(total_before):
            for j in range(length):
                for k in range(total_after):
                    input_idx = i * dim_size * total_after + (start + j) * total_after + k
                    output_idx = i * length * total_after + j * total_after + k
                    output_flat[output_idx] = input_flat[input_idx]
                    
        output = output_flat.reshape(output_shape)
    
    return output


def ones(shape, dtype=torch.float32, device=None):
    """Create a tensor filled with ones"""
    if device is None:
        device = torch.device("cpu")
    
    output = torch.empty(shape, dtype=dtype, device=device)
    n_elements = output.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    ones_kernel[grid](
        output.view(-1),
        n_elements,
        1.0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def ones_like(input_tensor):
    """Create a tensor of ones with the same shape and type as input"""
    return ones(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)


def zeros(shape, dtype=torch.float32, device=None):
    """Create a tensor filled with zeros"""
    if device is None:
        device = torch.device("cpu")
    
    output = torch.empty(shape, dtype=dtype, device=device)
    n_elements = output.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    ones_kernel[grid](
        output.view(-1),
        n_elements,
        0.0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def arange(start, end=None, step=1, dtype=torch.float32, device=None):
    """Create a tensor with a range of values"""
    if device is None:
        device = torch.device("cpu")
        
    if end is None:
        end = start
        start = 0
        
    n_elements = int((end - start) / step)
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


def repeat(input_tensor, repeats):
    """Repeat elements of a tensor along specified dimensions"""
    if isinstance(repeats, int):
        repeats = [repeats]
    
    # Calculate output shape
    input_shape = list(input_tensor.shape)
    if len(repeats) > len(input_shape):
        input_shape = [1] * (len(repeats) - len(input_shape)) + input_shape
    
    output_shape = []
    for i, (dim_size, repeat) in enumerate(zip_longest(input_shape, repeats, fillvalue=1)):
        output_shape.append(dim_size * (repeat if i < len(repeats) else 1))
    
    # Create output tensor and helper tensors
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    input_shape_tensor = torch.tensor(input_shape, dtype=torch.int32, device=input_tensor.device)
    output_shape_tensor = torch.tensor(output_shape, dtype=torch.int32, device=input_tensor.device)
    
    input_strides = torch.tensor([input_tensor.stride(i) for i in range(len(input_shape))],
                                dtype=torch.int32, device=input_tensor.device)
    output_strides = torch.tensor([output.stride(i) for i in range(len(output_shape))],
                                 dtype=torch.int32, device=input_tensor.device)
    
    # Launch kernel
    n_elements = output.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    repeat_kernel[grid](
        input_tensor.reshape(input_shape),
        output,
        input_shape_tensor,
        output_shape_tensor,
        input_strides,
        output_strides,
        len(input_shape),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def unsqueeze(input_tensor, dim):
    """Add a dimension of size 1 at the specified position"""
    shape = list(input_tensor.shape)
    
    if dim < 0:
        dim = len(shape) + 1 + dim
        
    assert 0 <= dim <= len(shape), f"dim {dim} out of range"
    
    new_shape = shape[:dim] + [1] + shape[dim:]
    return input_tensor.reshape(new_shape)


def transpose(input_tensor, dim0=0, dim1=1):
    """Transpose two dimensions of a tensor"""
    assert input_tensor.ndim >= 2, "Tensor must have at least 2 dimensions"
    
    if input_tensor.ndim == 2:
        M, N = input_tensor.shape
        output = torch.empty((N, M), dtype=input_tensor.dtype, device=input_tensor.device)
        
        BLOCK_SIZE = 16
        grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
        
        transpose_2d_kernel[grid](
            input_tensor, output,
            M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
    else:
        perm = list(range(input_tensor.ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return input_tensor.permute(perm)


def expand(input_tensor, shape):
    """Expand tensor to a new shape by broadcasting"""
    input_shape = list(input_tensor.shape)
    output_shape = list(shape)
    
    ndim = max(len(input_shape), len(output_shape))
    
    input_shape = [1] * (ndim - len(input_shape)) + input_shape
    output_shape = [1] * (ndim - len(output_shape)) + output_shape
    
    for i in range(ndim):
        if input_shape[i] != 1 and input_shape[i] != output_shape[i]:
            raise ValueError(f"Cannot expand dimension {i} from {input_shape[i]} to {output_shape[i]}")
    
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    input_shape_tensor = torch.tensor(input_shape, dtype=torch.int32, device=input_tensor.device)
    output_shape_tensor = torch.tensor(output_shape, dtype=torch.int32, device=input_tensor.device)
    
    input_strides = torch.tensor([input_tensor.stride(i) if i >= ndim - input_tensor.ndim else 0 
                                  for i in range(ndim)], dtype=torch.int32, device=input_tensor.device)
    output_strides = torch.tensor([output.stride(i) for i in range(ndim)], 
                                  dtype=torch.int32, device=input_tensor.device)
    
    n_elements = output.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    expand_kernel[grid](
        input_tensor.reshape(input_shape),
        output,
        input_shape_tensor,
        output_shape_tensor,
        input_strides,
        output_strides,
        ndim,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def reshape(input_tensor, shape):
    """Reshape tensor to a new shape"""
    return input_tensor.reshape(shape)


def triu(input_tensor, diagonal=0):
    """Return the upper triangular part of a matrix"""
    assert input_tensor.ndim == 2, "triu only supports 2D tensors"
    
    M, N = input_tensor.shape
    output = torch.zeros_like(input_tensor)
    
    BLOCK_SIZE = 256
    total_blocks = M * triton.cdiv(N, BLOCK_SIZE)
    grid = (total_blocks,)
    
    triu_kernel[grid](
        input_tensor,
        output,
        M, N,
        diagonal,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


# Test functions
def test_narrow():
    """Test narrow operation"""
    torch.manual_seed(42)
    
    x = torch.randn(10, 20, dtype=torch.float32, device=device)
    
    y_triton = narrow(x, dim=0, start=2, length=5)
    y_torch = torch.narrow(x, dim=0, start=2, length=5)
    
    torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)
    print(f"Narrow test (dim=0) passed! Shape: {y_triton.shape}")
    
    y_triton = narrow(x, dim=1, start=5, length=10)
    y_torch = torch.narrow(x, dim=1, start=5, length=10)
    
    torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)
    print(f"Narrow test (dim=1) passed! Shape: {y_triton.shape}")


def test_ones_zeros():
    """Test ones, ones_like, zeros operations"""
    torch.manual_seed(42)
    
    shape = (3, 4, 5)
    
    y_triton = ones(shape, dtype=torch.float32, device=device)
    y_torch = torch.ones(shape, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Ones test passed! Shape: {y_triton.shape}")
    
    x = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    y_triton = ones_like(x)
    y_torch = torch.ones_like(x)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Ones_like test passed! Shape: {y_triton.shape}")
    
    y_triton = zeros(shape, dtype=torch.float32, device=device)
    y_torch = torch.zeros(shape, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Zeros test passed! Shape: {y_triton.shape}")


def test_arange():
    """Test arange operation"""
    torch.manual_seed(42)
    
    y_triton = arange(10, dtype=torch.float32, device=device)
    y_torch = torch.arange(10, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Arange test (0-10) passed!")
    
    y_triton = arange(5, 15, 2, dtype=torch.float32, device=device)
    y_torch = torch.arange(5, 15, 2, dtype=torch.float32, device=device)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Arange test (5-15, step=2) passed!")


def test_repeat():
    """Test repeat operation"""
    torch.manual_seed(42)
    
    x = torch.randn(3, 4, dtype=torch.float32, device=device)
    
    y_triton = repeat(x, 2)
    y_torch = x.repeat(1, 2)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Repeat test passed! Shape: {x.shape} -> {y_triton.shape}")


def test_unsqueeze():
    """Test unsqueeze operation"""
    torch.manual_seed(42)
    
    x = torch.randn(3, 4, dtype=torch.float32, device=device)
    
    y_triton = unsqueeze(x, 0)
    y_torch = torch.unsqueeze(x, 0)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Unsqueeze test (dim=0) passed! Shape: {x.shape} -> {y_triton.shape}")
    
    y_triton = unsqueeze(x, 1)
    y_torch = torch.unsqueeze(x, 1)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Unsqueeze test (dim=1) passed! Shape: {x.shape} -> {y_triton.shape}")
    
    y_triton = unsqueeze(x, -1)
    y_torch = torch.unsqueeze(x, -1)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Unsqueeze test (dim=-1) passed! Shape: {x.shape} -> {y_triton.shape}")


def test_transpose():
    """Test transpose operation"""
    torch.manual_seed(42)
    
    x = torch.randn(3, 4, dtype=torch.float32, device=device)
    
    y_triton = transpose(x)
    y_torch = torch.transpose(x, 0, 1)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Transpose 2D test passed! Shape: {x.shape} -> {y_triton.shape}")
    
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32, device=device)
    y_triton = transpose(x, 1, 3)
    y_torch = torch.transpose(x, 1, 3)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Transpose 4D test passed! Shape: {x.shape} -> {y_triton.shape}")


def test_expand():
    """Test expand operation"""
    torch.manual_seed(42)
    
    x = torch.randn(1, 4, dtype=torch.float32, device=device)
    
    y_triton = expand(x, (3, 4))
    y_torch = x.expand(3, 4)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Expand test passed! Shape: {x.shape} -> {y_triton.shape}")
    
    x = torch.randn(3, 1, 5, dtype=torch.float32, device=device)
    y_triton = expand(x, (3, 4, 5))
    y_torch = x.expand(3, 4, 5)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Expand 3D test passed! Shape: {x.shape} -> {y_triton.shape}")


def test_reshape():
    """Test reshape operation"""
    torch.manual_seed(42)
    
    x = torch.randn(6, 4, dtype=torch.float32, device=device)
    
    y_triton = reshape(x, (3, 8))
    y_torch = x.reshape(3, 8)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Reshape test passed! Shape: {x.shape} -> {y_triton.shape}")
    
    y_triton = reshape(x, (-1, 12))
    y_torch = x.reshape(-1, 12)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Reshape with -1 test passed! Shape: {x.shape} -> {y_triton.shape}")


def test_triu():
    """Test upper triangular operation"""
    torch.manual_seed(42)
    
    x = torch.randn(5, 5, dtype=torch.float32, device=device)
    
    y_triton = triu(x)
    y_torch = torch.triu(x)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Triu test (diagonal=0) passed!")
    
    y_triton = triu(x, diagonal=1)
    y_torch = torch.triu(x, diagonal=1)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Triu test (diagonal=1) passed!")
    
    y_triton = triu(x, diagonal=-1)
    y_torch = torch.triu(x, diagonal=-1)
    torch.testing.assert_close(y_triton, y_torch)
    print(f"Triu test (diagonal=-1) passed!")


@pytest.mark.parametrize("shape, dim, start, length", [
    ((10, 20), 0, 2, 5),
    ((10, 20), 1, 5, 10),
    ((5, 10, 15), 2, 3, 7),
])
def test_narrow_parametrized(shape, dim, start, length):
    """Parametrized test for narrow"""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    
    y_triton = narrow(x, dim, start, length)
    y_torch = torch.narrow(x, dim, start, length)
    
    torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("start, end, step", [
    (0, 10, 1),
    (5, 15, 2),
    (0, 100, 5),
    (10, 50, 3),
])
def test_arange_parametrized(start, end, step):
    """Parametrized test for arange"""
    y_triton = arange(start, end, step, dtype=torch.float32, device=device)
    y_torch = torch.arange(start, end, step, dtype=torch.float32, device=device)
    
    torch.testing.assert_close(y_triton, y_torch)


@pytest.mark.parametrize("shape, new_shape", [
    ((1, 4), (3, 4)),
    ((3, 1, 5), (3, 4, 5)),
    ((1, 1, 7), (2, 3, 7)),
])
def test_expand_parametrized(shape, new_shape):
    """Parametrized test for expand"""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    
    y_triton = expand(x, new_shape)
    y_torch = x.expand(new_shape)
    
    torch.testing.assert_close(y_triton, y_torch)


if __name__ == "__main__":
    print("Testing Qwen3 Tensor Operations")
    print("=" * 50)
    
    print("\n1. Testing narrow...")
    test_narrow()
    
    print("\n2. Testing ones/zeros...")
    test_ones_zeros()
    
    print("\n3. Testing arange...")
    test_arange()
    
    print("\n4. Testing repeat...")
    test_repeat()
    
    print("\n5. Testing unsqueeze...")
    test_unsqueeze()
    
    print("\n6. Testing transpose...")
    test_transpose()
    
    print("\n7. Testing expand...")
    test_expand()
    
    print("\n8. Testing reshape...")
    test_reshape()
    
    print("\n9. Testing triu...")
    test_triu()
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
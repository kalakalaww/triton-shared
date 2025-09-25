module {
  func.func @fill_scalar_func_kernel_rank_1(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c128_i64 = arith.constant 128 : i64
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c128_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = tensor.empty() : tensor<128xi32>
    %4 = linalg.fill ins(%arg1 : i32) outs(%3 : tensor<128xi32>) -> tensor<128xi32>
    %5 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [128], strides: [1] : memref<*xi32> to memref<128xi32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast : (tensor<128xi32>, memref<128xi32, strided<[1], offset: ?>>) -> ()
    return
  }
}


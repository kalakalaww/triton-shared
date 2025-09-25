module {
  func.func @full_func_scalar_kernel_rank_1(%arg0: memref<*xi1> {tt.divisibility = 16 : i32}, %arg1: i1, %arg2: memref<*xi1> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c32_i64 = arith.constant 32 : i64
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c32_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = builtin.unrealized_conversion_cast %arg2 : memref<*xi1> to memref<*xi8>
    %4 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %3 to offset: [%4], sizes: [32], strides: [1] : memref<*xi8> to memref<32xi8, strided<[1], offset: ?>>
    %5 = arith.extui %arg1 : i1 to i8
    %6 = tensor.empty() : tensor<32xi8>
    %7 = linalg.fill ins(%5 : i8) outs(%6 : tensor<32xi8>) -> tensor<32xi8>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<32xi8>, memref<32xi8, strided<[1], offset: ?>>) -> ()
    return
  }
}


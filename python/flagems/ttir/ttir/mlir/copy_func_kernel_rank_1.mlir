module {
  func.func @copy_func_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c32_i64 = arith.constant 32 : i64
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c32_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<32xf32> to tensor<32xf32>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.muli %3, %5 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [32], strides: [%5] : memref<*xf32> to memref<32xf32, strided<[?], offset: ?>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<32xf32>, memref<32xf32, strided<[?], offset: ?>>) -> ()
    return
  }
}


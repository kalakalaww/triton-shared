module {
  func.func @embedding_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %0 = arith.extsi %arg6 : i32 to i64
    %1 = arith.muli %0, %c16_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = arith.index_cast %arg6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
    %4 = affine.load %reinterpret_cast[0] : memref<1xi32, strided<[1], offset: ?>>
    %5 = arith.muli %4, %c16_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<16xf32>
    memref.copy %reinterpret_cast_0, %alloc : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
    %7 = bufferization.to_tensor %alloc restrict writable : memref<16xf32> to tensor<16xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_1 : (tensor<16xf32>, memref<16xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


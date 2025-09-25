module {
  func.func @zeros_kernel(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : index
    %c1024_i64 = arith.constant 1024 : i64
    %0 = tensor.empty() : tensor<1024xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1024xi32>) -> tensor<1024xi32>
    %2 = arith.extsi %arg5 : i32 to i64
    %3 = arith.muli %2, %c1024_i64 : i64
    %4 = arith.index_cast %3 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %5 = arith.addi %4, %c1024 : index
    %6 = arith.index_cast %arg1 : i32 to index
    %7 = arith.minsi %5, %6 : index
    %8 = arith.maxsi %7, %4 : index
    %9 = arith.subi %8, %4 : index
    %extracted_slice = tensor.extract_slice %1[0] [%9] [1] : tensor<1024xi32> to tensor<?xi32>
    %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xi32>, memref<?xi32, strided<[1], offset: ?>>) -> ()
    return
  }
}


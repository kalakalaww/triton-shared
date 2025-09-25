module {
  func.func @diag_1d_to_2d_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c128_i64 = arith.constant 128 : i64
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = arith.extsi %arg7 : i32 to i64
    %1 = arith.muli %0, %c128_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %4 = arith.addi %2, %c128 : index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.minsi %4, %5 : index
    %7 = arith.maxsi %6, %2 : index
    %8 = arith.subi %7, %2 : index
    %alloc = memref.alloc() : memref<128xf32>
    %9 = arith.cmpi slt, %8, %c128 : index
    scf.if %9 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<128xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%8] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %10 = bufferization.to_tensor %alloc restrict writable : memref<128xf32> to tensor<128xf32>
    %11 = arith.muli %2, %3 : index
    %12 = arith.addi %11, %2 : index
    %13 = arith.addi %3, %c1 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [128], strides: [%13] : memref<*xf32> to memref<128xf32, strided<[?], offset: ?>>
    %14 = arith.minsi %4, %3 : index
    %15 = arith.maxsi %14, %2 : index
    %16 = arith.subi %15, %2 : index
    %extracted_slice = tensor.extract_slice %10[0] [%16] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_2 = memref.subview %reinterpret_cast_1[0] [%16] [1] : memref<128xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xf32>, memref<?xf32, strided<[?], offset: ?>>) -> ()
    return
  }
}


module {
  func.func @diag_2d_to_1d_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c128_i64 = arith.constant 128 : i64
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c128_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.muli %2, %3 : index
    %5 = arith.addi %4, %2 : index
    %6 = arith.addi %3, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [128], strides: [%6] : memref<*xf32> to memref<128xf32, strided<[?], offset: ?>>
    %7 = arith.addi %2, %c128 : index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.minsi %7, %8 : index
    %10 = arith.maxsi %9, %2 : index
    %11 = arith.subi %10, %2 : index
    %12 = arith.index_cast %arg3 : i32 to index
    %13 = arith.minsi %7, %12 : index
    %14 = arith.maxsi %13, %2 : index
    %15 = arith.subi %14, %2 : index
    %16 = arith.minsi %11, %15 : index
    %alloc = memref.alloc() : memref<128xf32>
    %17 = arith.cmpi slt, %16, %c128 : index
    scf.if %17 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<128xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%16] [1] : memref<128xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%16] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1]>>
    %18 = bufferization.to_tensor %alloc restrict writable : memref<128xf32> to tensor<128xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %18[0] [%16] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_2 = memref.subview %reinterpret_cast_1[0] [%16] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


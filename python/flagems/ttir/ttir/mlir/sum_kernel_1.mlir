module {
  func.func @sum_kernel_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c2_i64 = arith.constant 2 : i64
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.extsi %arg6 : i32 to i64
    %1 = arith.muli %0, %c2_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [2], strides: [1] : memref<*xf32> to memref<2xf32, strided<[1], offset: ?>>
    %3 = arith.addi %2, %c2 : index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.minsi %3, %4 : index
    %6 = arith.maxsi %5, %2 : index
    %7 = arith.subi %6, %2 : index
    %alloc = memref.alloc() : memref<2xf32>
    %8 = arith.cmpi slt, %7, %c2 : index
    scf.if %8 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<2xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<2xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%7] [1] : memref<2xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<2xf32> to tensor<2xf32>
    %10 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %10[] : tensor<f32>
    %reduced = linalg.reduce ins(%9 : tensor<2xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %11 = arith.index_cast %arg6 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %extracted, %reinterpret_cast_1[0] : memref<1xf32, strided<[1], offset: ?>>
    return
  }
}


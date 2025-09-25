#map = affine_map<(d0) -> (d0)>
module {
  func.func @dot_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0 = arith.constant 0 : index
    %c1024_i64 = arith.constant 1024 : i64
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.extsi %arg7 : i32 to i64
    %1 = arith.muli %0, %c1024_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %3 = arith.addi %2, %c1024 : index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.minsi %3, %4 : index
    %6 = arith.maxsi %5, %2 : index
    %7 = arith.subi %6, %2 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %8 = arith.cmpi slt, %7, %c1024 : index
    scf.if %8 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%7] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    scf.if %8 {
      linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1024xf32>)
    }
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%7] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%7] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32> to tensor<1024xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %10 : tensor<1024xf32>, tensor<1024xf32>) outs(%9 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %13 = arith.mulf %in, %in_6 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    %12 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %12[] : tensor<f32>
    %reduced = linalg.reduce ins(%11 : tensor<1024xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %extracted, %reinterpret_cast_5[0] : memref<1xf32, strided<[1], offset: ?>>
    return
  }
}


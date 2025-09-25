#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @weight_norm_kernel_first(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = tensor.empty() : tensor<32x2048xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x2048xf32>) -> tensor<32x2048xf32>
    %2 = arith.extsi %arg10 : i32 to i64
    %3 = arith.muli %2, %c32_i64 : i64
    %4 = arith.index_cast %3 : i64 to index
    %5 = arith.index_cast %arg5 : i32 to index
    %6 = arith.muli %4, %5 : index
    %7 = scf.for %arg13 = %c0_i32 to %arg5 step %c2048_i32 iter_args(%arg14 = %1) -> (tensor<32x2048xf32>)  : i32 {
      %22 = arith.index_cast %arg13 : i32 to index
      %23 = arith.addi %6, %22 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [32, 2048], strides: [%5, 1] : memref<*xf32> to memref<32x2048xf32, strided<[?, 1], offset: ?>>
      %24 = arith.addi %22, %c2048 : index
      %25 = arith.minsi %24, %5 : index
      %26 = arith.maxsi %25, %22 : index
      %27 = arith.subi %26, %22 : index
      %28 = arith.addi %4, %c32 : index
      %29 = arith.index_cast %arg4 : i32 to index
      %30 = arith.minsi %28, %29 : index
      %31 = arith.maxsi %30, %4 : index
      %32 = arith.subi %31, %4 : index
      %33 = arith.minsi %32, %c32 : index
      %34 = arith.minsi %27, %c2048 : index
      %alloc_4 = memref.alloc() : memref<32x2048xf32>
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%33, %34] [1, 1] : memref<32x2048xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[0, 0] [%33, %34] [1, 1] : memref<32x2048xf32> to memref<?x?xf32, strided<[2048, 1]>>
      memref.copy %subview_5, %subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[2048, 1]>>
      %35 = bufferization.to_tensor %alloc_4 restrict writable : memref<32x2048xf32> to tensor<32x2048xf32>
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%35, %35 : tensor<32x2048xf32>, tensor<32x2048xf32>) outs(%35 : tensor<32x2048xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %38 = arith.mulf %in, %in_7 : f32
        linalg.yield %38 : f32
      } -> tensor<32x2048xf32>
      %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<32x2048xf32>, tensor<32x2048xf32>) outs(%arg14 : tensor<32x2048xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %38 = arith.addf %in, %in_7 : f32
        linalg.yield %38 : f32
      } -> tensor<32x2048xf32>
      scf.yield %37 : tensor<32x2048xf32>
    }
    %8 = tensor.empty() : tensor<2048x32xf32>
    %transposed = linalg.transpose ins(%7 : tensor<32x2048xf32>) outs(%8 : tensor<2048x32xf32>) permutation = [1, 0] 
    %9 = tensor.empty() : tensor<32xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce ins(%transposed : tensor<2048x32xf32>) outs(%10 : tensor<32xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %22 = arith.addf %in, %init : f32
        linalg.yield %22 : f32
      }
    %11 = linalg.fill ins(%arg6 : f32) outs(%9 : tensor<32xf32>) -> tensor<32xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%reduced, %11 : tensor<32xf32>, tensor<32xf32>) outs(%reduced : tensor<32xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %22 = arith.addf %in, %in_3 : f32
      linalg.yield %22 : f32
    } -> tensor<32xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%12 : tensor<32xf32>) outs(%12 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.sqrt %in : f32
      linalg.yield %22 : f32
    } -> tensor<32xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [32, 1], strides: [1, 1] : memref<*xf32> to memref<32x1xf32, strided<[1, 1], offset: ?>>
    %expanded = tensor.expand_shape %13 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %14 = arith.addi %4, %c32 : index
    %15 = arith.index_cast %arg4 : i32 to index
    %16 = arith.minsi %14, %15 : index
    %17 = arith.maxsi %16, %4 : index
    %18 = arith.subi %17, %4 : index
    %extracted_slice = tensor.extract_slice %expanded[0, 0] [%18, 1] [1, 1] : tensor<32x1xf32> to tensor<?x1xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%18, 1] [1, 1] : memref<32x1xf32, strided<[1, 1], offset: ?>> to memref<?x1xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x1xf32>, memref<?x1xf32, strided<[1, 1], offset: ?>>) -> ()
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [32, 1], strides: [1, 1] : memref<*xf32> to memref<32x1xf32, strided<[1, 1], offset: ?>>
    %alloc = memref.alloc() : memref<32x1xf32>
    %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [%18, 1] [1, 1] : memref<32x1xf32, strided<[1, 1], offset: ?>> to memref<?x1xf32, strided<[1, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0] [%18, 1] [1, 1] : memref<32x1xf32> to memref<?x1xf32, strided<[1, 1]>>
    memref.copy %subview_1, %subview_2 : memref<?x1xf32, strided<[1, 1], offset: ?>> to memref<?x1xf32, strided<[1, 1]>>
    %19 = bufferization.to_tensor %alloc restrict writable : memref<32x1xf32> to tensor<32x1xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<32x1xf32>) outs(%0 : tensor<32x2048xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2048xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<32x1xf32>) outs(%0 : tensor<32x2048xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2048xf32>
    scf.for %arg13 = %c0_i32 to %arg5 step %c2048_i32  : i32 {
      %22 = arith.index_cast %arg13 : i32 to index
      %23 = arith.addi %6, %22 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [32, 2048], strides: [%5, 1] : memref<*xf32> to memref<32x2048xf32, strided<[?, 1], offset: ?>>
      %24 = arith.addi %22, %c2048 : index
      %25 = arith.minsi %24, %5 : index
      %26 = arith.maxsi %25, %22 : index
      %27 = arith.subi %26, %22 : index
      %28 = arith.minsi %18, %c32 : index
      %29 = arith.minsi %27, %c2048 : index
      %alloc_4 = memref.alloc() : memref<32x2048xf32>
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%28, %29] [1, 1] : memref<32x2048xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[0, 0] [%28, %29] [1, 1] : memref<32x2048xf32> to memref<?x?xf32, strided<[2048, 1]>>
      memref.copy %subview_5, %subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[2048, 1]>>
      %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<32x2048xf32> to tensor<32x2048xf32>
      %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%30, %20 : tensor<32x2048xf32>, tensor<32x2048xf32>) outs(%30 : tensor<32x2048xf32>) {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %33 = arith.divf %in, %in_10 : f32
        linalg.yield %33 : f32
      } -> tensor<32x2048xf32>
      %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%31, %21 : tensor<32x2048xf32>, tensor<32x2048xf32>) outs(%31 : tensor<32x2048xf32>) {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %33 = arith.mulf %in, %in_10 : f32
        linalg.yield %33 : f32
      } -> tensor<32x2048xf32>
      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 2048], strides: [%5, 1] : memref<*xf32> to memref<32x2048xf32, strided<[?, 1], offset: ?>>
      %extracted_slice_8 = tensor.extract_slice %32[0, 0] [%28, %29] [1, 1] : tensor<32x2048xf32> to tensor<?x?xf32>
      %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%28, %29] [1, 1] : memref<32x2048xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_8 in writable %subview_9 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}


#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_func_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c4_i64 = arith.constant 4 : i64
    %0 = arith.extsi %arg9 : i32 to i64
    %1 = arith.muli %0, %c4_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<4xf32> to tensor<4xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<4xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<4xf32> to tensor<4xf32>
    %6 = arith.sitofp %arg2 : i32 to f32
    %7 = tensor.empty() : tensor<4xf32>
    %8 = linalg.fill ins(%6 : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%5, %8 : tensor<4xf32>, tensor<4xf32>) outs(%5 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %11 = arith.mulf %in, %in_3 : f32
      linalg.yield %11 : f32
    } -> tensor<4xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %9 : tensor<4xf32>, tensor<4xf32>) outs(%4 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %11 = arith.addf %in, %in_3 : f32
      linalg.yield %11 : f32
    } -> tensor<4xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast_2 : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


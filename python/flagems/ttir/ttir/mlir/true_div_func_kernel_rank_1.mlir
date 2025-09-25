#map = affine_map<(d0) -> (d0)>
module {
  func.func @true_div_func_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = arith.index_cast %arg6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<1xf32> to tensor<1xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
    %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<1xf32> to tensor<1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %2 : tensor<1xf32>, tensor<1xf32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %4 = arith.divf %in, %in_3 : f32
      linalg.yield %4 : f32
    } -> tensor<1xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


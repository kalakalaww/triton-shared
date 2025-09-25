#map = affine_map<(d0) -> (d0)>
module {
  func.func @abs_func_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c4_i64 = arith.constant 4 : i64
    %0 = arith.extsi %arg7 : i32 to i64
    %1 = arith.muli %0, %c4_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<4xf32> to tensor<4xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<4xf32>) outs(%4 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.absf %in : f32
      linalg.yield %6 : f32
    } -> tensor<4xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_0 : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


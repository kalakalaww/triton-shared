#map = affine_map<(d0) -> (d0)>
module {
  func.func @rsqrt_func_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %cst = arith.constant 1.000000e+00 : f32
    %c512_i64 = arith.constant 512 : i64
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %2 = arith.extsi %arg7 : i32 to i64
    %3 = arith.muli %2, %c512_i64 : i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = arith.index_cast %4 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<512xf32>
    memref.copy %reinterpret_cast, %alloc : memref<512xf32, strided<[1], offset: ?>> to memref<512xf32>
    %6 = bufferization.to_tensor %alloc restrict writable : memref<512xf32> to tensor<512xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%6 : tensor<512xf32>) outs(%6 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.sqrt %in : f32
      linalg.yield %9 : f32
    } -> tensor<512xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %7 : tensor<512xf32>, tensor<512xf32>) outs(%1 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.divf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<512xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %8 in writable %reinterpret_cast_0 : (tensor<512xf32>, memref<512xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


#map = affine_map<(d0) -> (d0)>
module {
  func.func @angle_float_and_int_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.14159274 : f32
    %c512_i64 = arith.constant 512 : i64
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = arith.muli %3, %c512_i64 : i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.index_cast %5 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<512xf32>
    memref.copy %reinterpret_cast, %alloc : memref<512xf32, strided<[1], offset: ?>> to memref<512xf32>
    %7 = bufferization.to_tensor %alloc restrict writable : memref<512xf32> to tensor<512xf32>
    %8 = tensor.empty() : tensor<512xi1>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %1 : tensor<512xf32>, tensor<512xf32>) outs(%8 : tensor<512xi1>) {
    ^bb0(%in: f32, %in_2: f32, %out: i1):
      %11 = arith.cmpf oge, %in, %in_2 : f32
      linalg.yield %11 : i1
    } -> tensor<512xi1>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%9, %1, %2 : tensor<512xi1>, tensor<512xf32>, tensor<512xf32>) outs(%1 : tensor<512xf32>) {
    ^bb0(%in: i1, %in_2: f32, %in_3: f32, %out: f32):
      %11 = arith.select %in, %in_2, %in_3 : f32
      linalg.yield %11 : f32
    } -> tensor<512xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast_1 : (tensor<512xf32>, memref<512xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


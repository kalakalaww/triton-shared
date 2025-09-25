#map = affine_map<(d0) -> (d0)>
module {
  func.func @eq_func_scalar_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: f32, %arg2: memref<*xi1> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c128_i64 = arith.constant 128 : i64
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c128_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<128xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<128xf32> to tensor<128xf32>
    %5 = tensor.empty() : tensor<128xf32>
    %6 = linalg.fill ins(%arg1 : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
    %7 = tensor.empty() : tensor<128xi1>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %6 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xi1>) {
    ^bb0(%in: f32, %in_1: f32, %out: i1):
      %12 = arith.cmpf oeq, %in, %in_1 : f32
      linalg.yield %12 : i1
    } -> tensor<128xi1>
    %9 = builtin.unrealized_conversion_cast %arg2 : memref<*xi1> to memref<*xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %9 to offset: [%3], sizes: [128], strides: [1] : memref<*xi8> to memref<128xi8, strided<[1], offset: ?>>
    %10 = tensor.empty() : tensor<128xi8>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%8 : tensor<128xi1>) outs(%10 : tensor<128xi8>) {
    ^bb0(%in: i1, %out: i8):
      %12 = arith.extui %in : i1 to i8
      linalg.yield %12 : i8
    } -> tensor<128xi8>
    bufferization.materialize_in_destination %11 in writable %reinterpret_cast_0 : (tensor<128xi8>, memref<128xi8, strided<[1], offset: ?>>) -> ()
    return
  }
}


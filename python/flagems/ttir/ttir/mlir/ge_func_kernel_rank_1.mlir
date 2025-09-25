#map = affine_map<(d0) -> (d0)>
module {
  func.func @ge_func_kernel_rank_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xi1> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c32_i64 = arith.constant 32 : i64
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c32_i64 : i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<32xf32> to tensor<32xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<32xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<32xf32> to tensor<32xf32>
    %6 = tensor.empty() : tensor<32xi1>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %5 : tensor<32xf32>, tensor<32xf32>) outs(%6 : tensor<32xi1>) {
    ^bb0(%in: f32, %in_3: f32, %out: i1):
      %11 = arith.cmpf oge, %in, %in_3 : f32
      linalg.yield %11 : i1
    } -> tensor<32xi1>
    %8 = builtin.unrealized_conversion_cast %arg2 : memref<*xi1> to memref<*xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %8 to offset: [%3], sizes: [32], strides: [1] : memref<*xi8> to memref<32xi8, strided<[1], offset: ?>>
    %9 = tensor.empty() : tensor<32xi8>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%7 : tensor<32xi1>) outs(%9 : tensor<32xi8>) {
    ^bb0(%in: i1, %out: i8):
      %11 = arith.extui %in : i1 to i8
      linalg.yield %11 : i8
    } -> tensor<32xi8>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast_2 : (tensor<32xi8>, memref<32xi8, strided<[1], offset: ?>>) -> ()
    return
  }
}


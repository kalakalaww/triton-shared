#map = affine_map<(d0) -> (d0)>
module {
  func.func @arange_func(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c128_i64 = arith.constant 128 : i64
    %c128 = arith.constant 128 : index
    %0 = arith.extsi %arg7 : i32 to i64
    %1 = arith.muli %0, %c128_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = tensor.empty() : tensor<128xi32>
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%3 : tensor<128xi32>) {
    ^bb0(%out: i32):
      %19 = linalg.index 0 : index
      %20 = arith.index_cast %19 : index to i32
      linalg.yield %20 : i32
    } -> tensor<128xi32>
    %5 = tensor.empty() : tensor<128xi64>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<128xi32>) outs(%5 : tensor<128xi64>) {
    ^bb0(%in: i32, %out: i64):
      %19 = arith.extsi %in : i32 to i64
      linalg.yield %19 : i64
    } -> tensor<128xi64>
    %7 = linalg.fill ins(%1 : i64) outs(%5 : tensor<128xi64>) -> tensor<128xi64>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %7 : tensor<128xi64>, tensor<128xi64>) outs(%6 : tensor<128xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %19 = arith.addi %in, %in_0 : i64
      linalg.yield %19 : i64
    } -> tensor<128xi64>
    %9 = arith.extsi %arg1 : i32 to i64
    %10 = linalg.fill ins(%9 : i64) outs(%5 : tensor<128xi64>) -> tensor<128xi64>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %10 : tensor<128xi64>, tensor<128xi64>) outs(%8 : tensor<128xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %19 = arith.addi %in, %in_0 : i64
      linalg.yield %19 : i64
    } -> tensor<128xi64>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
    %12 = tensor.empty() : tensor<128xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%11 : tensor<128xi64>) outs(%12 : tensor<128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %19 = arith.sitofp %in : i64 to f32
      linalg.yield %19 : f32
    } -> tensor<128xf32>
    %14 = arith.addi %2, %c128 : index
    %15 = arith.index_cast %arg3 : i32 to index
    %16 = arith.minsi %14, %15 : index
    %17 = arith.maxsi %16, %2 : index
    %18 = arith.subi %17, %2 : index
    %extracted_slice = tensor.extract_slice %13[0] [%18] [1] : tensor<128xf32> to tensor<?xf32>
    %subview = memref.subview %reinterpret_cast[0] [%18] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


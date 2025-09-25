#map = affine_map<(d0) -> (d0)>
module {
  func.func @var_mean_kernel_2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<1xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1xi32>) -> tensor<1xi32>
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %2 = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
    %3 = tensor.empty() : tensor<1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<1xi32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%in: i32, %out: f32):
      %13 = arith.index_cast %in : i32 to index
      %extracted = tensor.extract %2[%13] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1xf32>
    %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %5 = bufferization.to_tensor %cast_0 restrict : memref<?xf32> to tensor<?xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<1xi32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%in: i32, %out: f32):
      %13 = arith.index_cast %in : i32 to index
      %extracted = tensor.extract %5[%13] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1xf32>
    %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %7 = bufferization.to_tensor %cast_1 restrict : memref<?xf32> to tensor<?xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<1xi32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%in: i32, %out: f32):
      %13 = arith.index_cast %in : i32 to index
      %extracted = tensor.extract %7[%13] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1xf32>
    %9:3 = "tt.reduce"(%6, %8, %4) <{axis = 0 : i32}> ({
    ^bb0(%arg13: f32, %arg14: f32, %arg15: f32, %arg16: f32, %arg17: f32, %arg18: f32):
      %13 = arith.addf %arg14, %arg17 : f32
      %14 = arith.maxnumf %13, %cst : f32
      %15 = arith.mulf %arg13, %arg14 : f32
      %16 = arith.mulf %arg16, %arg17 : f32
      %17 = arith.addf %15, %16 : f32
      %18 = arith.divf %17, %14 : f32
      %19 = arith.mulf %15, %arg13 : f32
      %20 = arith.addf %arg15, %19 : f32
      %21 = arith.addf %20, %arg18 : f32
      %22 = arith.mulf %16, %arg16 : f32
      %23 = arith.addf %21, %22 : f32
      %24 = arith.mulf %13, %18 : f32
      %25 = arith.mulf %24, %18 : f32
      %26 = arith.subf %23, %25 : f32
      tt.reduce.return %18, %13, %26 : f32, f32, f32
    }) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> (f32, f32, f32)
    %10 = arith.sitofp %arg5 : i32 to f32
    %11 = arith.subf %10, %arg6 : f32
    %12 = arith.divf %9#2, %11 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %9#0, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%c0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %12, %reinterpret_cast_2[0] : memref<1xf32, strided<[1], offset: ?>>
    return
  }
}


#map = affine_map<(d0) -> (d0)>
module {
  func.func @upsample_nearest2d_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: f32, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.extsi %arg12 : i32 to i64
    %1 = arith.muli %0, %c2048_i64 : i64
    %2 = tensor.empty() : tensor<2048xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<2048xi32>) {
    ^bb0(%out: i32):
      %50 = linalg.index 0 : index
      %51 = arith.index_cast %50 : index to i32
      linalg.yield %51 : i32
    } -> tensor<2048xi32>
    %4 = tensor.empty() : tensor<2048xi64>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : tensor<2048xi32>) outs(%4 : tensor<2048xi64>) {
    ^bb0(%in: i32, %out: i64):
      %50 = arith.extsi %in : i32 to i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %6 = linalg.fill ins(%1 : i64) outs(%4 : tensor<2048xi64>) -> tensor<2048xi64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %5 : tensor<2048xi64>, tensor<2048xi64>) outs(%6 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.addi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %8 = arith.extsi %arg4 : i32 to i64
    %9 = linalg.fill ins(%8 : i64) outs(%4 : tensor<2048xi64>) -> tensor<2048xi64>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %9 : tensor<2048xi64>, tensor<2048xi64>) outs(%7 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.remsi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %9 : tensor<2048xi64>, tensor<2048xi64>) outs(%7 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.divsi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %12 = arith.extsi %arg3 : i32 to i64
    %13 = linalg.fill ins(%12 : i64) outs(%4 : tensor<2048xi64>) -> tensor<2048xi64>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%11, %13 : tensor<2048xi64>, tensor<2048xi64>) outs(%11 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.remsi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%11, %13 : tensor<2048xi64>, tensor<2048xi64>) outs(%11 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.divsi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %16 = arith.extsi %arg2 : i32 to i64
    %17 = linalg.fill ins(%16 : i64) outs(%4 : tensor<2048xi64>) -> tensor<2048xi64>
    %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%15, %17 : tensor<2048xi64>, tensor<2048xi64>) outs(%15 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.remsi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %19 = tensor.empty() : tensor<2048xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<2048xi64>) outs(%19 : tensor<2048xf32>) {
    ^bb0(%in: i64, %out: f32):
      %50 = arith.sitofp %in : i64 to f32
      linalg.yield %50 : f32
    } -> tensor<2048xf32>
    %21 = linalg.fill ins(%arg7 : f32) outs(%19 : tensor<2048xf32>) -> tensor<2048xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%20, %21 : tensor<2048xf32>, tensor<2048xf32>) outs(%20 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %50 = arith.mulf %in, %in_1 : f32
      linalg.yield %50 : f32
    } -> tensor<2048xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%22 : tensor<2048xf32>) outs(%2 : tensor<2048xi32>) {
    ^bb0(%in: f32, %out: i32):
      %50 = arith.fptosi %in : f32 to i32
      linalg.yield %50 : i32
    } -> tensor<2048xi32>
    %24 = arith.subi %arg5, %c1_i32 : i32
    %25 = linalg.fill ins(%24 : i32) outs(%2 : tensor<2048xi32>) -> tensor<2048xi32>
    %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%23, %25 : tensor<2048xi32>, tensor<2048xi32>) outs(%23 : tensor<2048xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %50 = arith.minsi %in, %in_1 : i32
      linalg.yield %50 : i32
    } -> tensor<2048xi32>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : tensor<2048xi64>) outs(%19 : tensor<2048xf32>) {
    ^bb0(%in: i64, %out: f32):
      %50 = arith.sitofp %in : i64 to f32
      linalg.yield %50 : f32
    } -> tensor<2048xf32>
    %28 = linalg.fill ins(%arg8 : f32) outs(%19 : tensor<2048xf32>) -> tensor<2048xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%27, %28 : tensor<2048xf32>, tensor<2048xf32>) outs(%27 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %50 = arith.mulf %in, %in_1 : f32
      linalg.yield %50 : f32
    } -> tensor<2048xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%29 : tensor<2048xf32>) outs(%2 : tensor<2048xi32>) {
    ^bb0(%in: f32, %out: i32):
      %50 = arith.fptosi %in : f32 to i32
      linalg.yield %50 : i32
    } -> tensor<2048xi32>
    %31 = arith.subi %arg6, %c1_i32 : i32
    %32 = linalg.fill ins(%31 : i32) outs(%2 : tensor<2048xi32>) -> tensor<2048xi32>
    %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%30, %32 : tensor<2048xi32>, tensor<2048xi32>) outs(%30 : tensor<2048xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %50 = arith.minsi %in, %in_1 : i32
      linalg.yield %50 : i32
    } -> tensor<2048xi32>
    %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%18, %13 : tensor<2048xi64>, tensor<2048xi64>) outs(%18 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.muli %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%34, %14 : tensor<2048xi64>, tensor<2048xi64>) outs(%34 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.addi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%35, %9 : tensor<2048xi64>, tensor<2048xi64>) outs(%35 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.muli %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%36, %10 : tensor<2048xi64>, tensor<2048xi64>) outs(%36 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.addi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %38 = arith.extsi %arg5 : i32 to i64
    %39 = linalg.fill ins(%38 : i64) outs(%4 : tensor<2048xi64>) -> tensor<2048xi64>
    %40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%18, %39 : tensor<2048xi64>, tensor<2048xi64>) outs(%18 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.muli %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %41 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%26 : tensor<2048xi32>) outs(%4 : tensor<2048xi64>) {
    ^bb0(%in: i32, %out: i64):
      %50 = arith.extsi %in : i32 to i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%40, %41 : tensor<2048xi64>, tensor<2048xi64>) outs(%40 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.addi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %43 = arith.extsi %arg6 : i32 to i64
    %44 = linalg.fill ins(%43 : i64) outs(%4 : tensor<2048xi64>) -> tensor<2048xi64>
    %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%42, %44 : tensor<2048xi64>, tensor<2048xi64>) outs(%42 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.muli %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %46 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%33 : tensor<2048xi32>) outs(%4 : tensor<2048xi64>) {
    ^bb0(%in: i32, %out: i64):
      %50 = arith.extsi %in : i32 to i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%45, %46 : tensor<2048xi64>, tensor<2048xi64>) outs(%45 : tensor<2048xi64>) {
    ^bb0(%in: i64, %in_1: i64, %out: i64):
      %50 = arith.addi %in, %in_1 : i64
      linalg.yield %50 : i64
    } -> tensor<2048xi64>
    %cast = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %48 = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
    %49 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%47 : tensor<2048xi64>) outs(%19 : tensor<2048xf32>) {
    ^bb0(%in: i64, %out: f32):
      %50 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %48[%50] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<2048xf32>
    %cast_0 = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%37, %49 : tensor<2048xi64>, tensor<2048xf32>) {
    ^bb0(%in: i64, %in_1: f32):
      %50 = arith.index_cast %in : i64 to index
      memref.store %in_1, %cast_0[%50] : memref<?xf32>
      linalg.yield
    }
    return
  }
}


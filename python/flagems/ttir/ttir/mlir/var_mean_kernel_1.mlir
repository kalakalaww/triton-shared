#map = affine_map<(d0) -> (d0)>
module {
  func.func @var_mean_kernel_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c1024_i64 = arith.constant 1024 : i64
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.extsi %arg8 : i32 to i64
    %1 = arith.muli %0, %c1024_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = tensor.empty() : tensor<1024xi32>
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%3 : tensor<1024xi32>) {
    ^bb0(%out: i32):
      %31 = linalg.index 0 : index
      %32 = arith.index_cast %31 : index to i32
      linalg.yield %32 : i32
    } -> tensor<1024xi32>
    %5 = tensor.empty() : tensor<1024xi64>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<1024xi32>) outs(%5 : tensor<1024xi64>) {
    ^bb0(%in: i32, %out: i64):
      %31 = arith.extsi %in : i32 to i64
      linalg.yield %31 : i64
    } -> tensor<1024xi64>
    %7 = linalg.fill ins(%1 : i64) outs(%5 : tensor<1024xi64>) -> tensor<1024xi64>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %6 : tensor<1024xi64>, tensor<1024xi64>) outs(%7 : tensor<1024xi64>) {
    ^bb0(%in: i64, %in_10: i64, %out: i64):
      %31 = arith.addi %in, %in_10 : i64
      linalg.yield %31 : i64
    } -> tensor<1024xi64>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %9 = arith.extsi %arg4 : i32 to i64
    %10 = linalg.fill ins(%9 : i64) outs(%5 : tensor<1024xi64>) -> tensor<1024xi64>
    %11 = tensor.empty() : tensor<1024xi1>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %10 : tensor<1024xi64>, tensor<1024xi64>) outs(%11 : tensor<1024xi1>) {
    ^bb0(%in: i64, %in_10: i64, %out: i1):
      %31 = arith.cmpi slt, %in, %in_10 : i64
      linalg.yield %31 : i1
    } -> tensor<1024xi1>
    %13 = arith.addi %2, %c1024 : index
    %14 = arith.index_cast %arg4 : i32 to index
    %15 = arith.minsi %13, %14 : index
    %16 = arith.maxsi %15, %2 : index
    %17 = arith.subi %16, %2 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %18 = arith.cmpi slt, %17, %c1024 : index
    scf.if %18 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%17] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%17] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %19 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %20 = tensor.empty() : tensor<1024xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<1024xi1>) outs(%20 : tensor<1024xf32>) {
    ^bb0(%in: i1, %out: f32):
      %31 = arith.uitofp %in : i1 to f32
      linalg.yield %31 : f32
    } -> tensor<1024xf32>
    %22 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %22[] : tensor<f32>
    %reduced = linalg.reduce ins(%21 : tensor<1024xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %23 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_1 = tensor.insert %cst into %23[] : tensor<f32>
    %reduced_2 = linalg.reduce ins(%19 : tensor<1024xf32>) outs(%inserted_1 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %extracted_3 = tensor.extract %reduced_2[] : tensor<f32>
    %24 = arith.divf %extracted_3, %extracted : f32
    %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%19, %19 : tensor<1024xf32>, tensor<1024xf32>) outs(%19 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %31 = arith.mulf %in, %in_10 : f32
      linalg.yield %31 : f32
    } -> tensor<1024xf32>
    %26 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_4 = tensor.insert %cst into %26[] : tensor<f32>
    %reduced_5 = linalg.reduce ins(%25 : tensor<1024xf32>) outs(%inserted_4 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %extracted_6 = tensor.extract %reduced_5[] : tensor<f32>
    %27 = arith.mulf %extracted, %24 : f32
    %28 = arith.mulf %27, %24 : f32
    %29 = arith.subf %extracted_6, %28 : f32
    %30 = arith.index_cast %arg8 : i32 to index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %24, %reinterpret_cast_7[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %29, %reinterpret_cast_8[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %extracted, %reinterpret_cast_9[0] : memref<1xf32, strided<[1], offset: ?>>
    return
  }
}


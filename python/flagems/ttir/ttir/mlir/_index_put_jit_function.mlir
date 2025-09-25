#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @_index_put_jit_function(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: memref<*xi32> {tt.divisibility = 16 : i32}, %arg4: memref<*xi32> {tt.divisibility = 16 : i32}, %arg5: memref<*xf32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i64 = arith.constant 4 : i64
    %0 = tensor.empty() : tensor<4x1xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %2 = arith.extsi %arg22 : i32 to i64
    %3 = arith.extsi %arg23 : i32 to i64
    %4 = arith.muli %2, %c4_i64 : i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = tensor.empty() : tensor<4xi32>
    %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<4xi32>) {
    ^bb0(%out: i32):
      %72 = linalg.index 0 : index
      %73 = arith.index_cast %72 : index to i32
      linalg.yield %73 : i32
    } -> tensor<4xi32>
    %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
    %8 = tensor.empty() : tensor<4x1xi64>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<4x1xi32>) outs(%8 : tensor<4x1xi64>) {
    ^bb0(%in: i32, %out: i64):
      %72 = arith.extsi %in : i32 to i64
      linalg.yield %72 : i64
    } -> tensor<4x1xi64>
    %10 = linalg.fill ins(%4 : i64) outs(%8 : tensor<4x1xi64>) -> tensor<4x1xi64>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10, %9 : tensor<4x1xi64>, tensor<4x1xi64>) outs(%10 : tensor<4x1xi64>) {
    ^bb0(%in: i64, %in_26: i64, %out: i64):
      %72 = arith.addi %in, %in_26 : i64
      linalg.yield %72 : i64
    } -> tensor<4x1xi64>
    %12 = arith.extsi %arg10 : i32 to i64
    %13 = arith.index_cast %arg10 : i32 to index
    %14 = linalg.fill ins(%12 : i64) outs(%8 : tensor<4x1xi64>) -> tensor<4x1xi64>
    %15 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11, %14 : tensor<4x1xi64>, tensor<4x1xi64>) outs(%11 : tensor<4x1xi64>) {
    ^bb0(%in: i64, %in_26: i64, %out: i64):
      %72 = arith.remsi %in, %in_26 : i64
      linalg.yield %72 : i64
    } -> tensor<4x1xi64>
    %16 = arith.extsi %arg18 : i32 to i64
    %17 = linalg.fill ins(%16 : i64) outs(%8 : tensor<4x1xi64>) -> tensor<4x1xi64>
    %18 = tensor.empty() : tensor<4x1xi1>
    %19 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11, %17 : tensor<4x1xi64>, tensor<4x1xi64>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i64, %in_26: i64, %out: i1):
      %72 = arith.cmpi slt, %in, %in_26 : i64
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %20 = arith.subi %13, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [%20, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %21 = arith.subi %c4, %20 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [%21, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %22 = arith.addi %5, %c4 : index
    %23 = arith.index_cast %arg18 : i32 to index
    %24 = arith.minsi %22, %23 : index
    %25 = arith.maxsi %24, %5 : index
    %26 = arith.subi %25, %5 : index
    %alloc = memref.alloc() : memref<4x1xi32>
    %27 = arith.cmpi slt, %26, %c4 : index
    scf.if %27 {
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x1xi32>)
    }
    %28 = arith.minsi %20, %26 : index
    %29 = arith.subi %26, %28 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%28, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [%29, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0] [%28, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1]>>
    %subview_3 = memref.subview %alloc[%28, 0] [%29, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    memref.copy %subview, %subview_2 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1]>>
    memref.copy %subview_1, %subview_3 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    %30 = bufferization.to_tensor %alloc restrict writable : memref<4x1xi32> to tensor<4x1xi32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [%20, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [%21, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %alloc_6 = memref.alloc() : memref<4x1xi32>
    scf.if %27 {
      linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<4x1xi32>)
    }
    %subview_7 = memref.subview %reinterpret_cast_4[0, 0] [%28, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_8 = memref.subview %reinterpret_cast_5[0, 0] [%29, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_9 = memref.subview %alloc_6[0, 0] [%28, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1]>>
    %subview_10 = memref.subview %alloc_6[%28, 0] [%29, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    memref.copy %subview_7, %subview_9 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1]>>
    memref.copy %subview_8, %subview_10 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    %31 = bufferization.to_tensor %alloc_6 restrict writable : memref<4x1xi32> to tensor<4x1xi32>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%5], sizes: [%20, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%c0], sizes: [%21, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %alloc_13 = memref.alloc() : memref<4x1xi32>
    scf.if %27 {
      linalg.fill ins(%c0_i32 : i32) outs(%alloc_13 : memref<4x1xi32>)
    }
    %subview_14 = memref.subview %reinterpret_cast_11[0, 0] [%28, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_15 = memref.subview %reinterpret_cast_12[0, 0] [%29, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_16 = memref.subview %alloc_13[0, 0] [%28, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1]>>
    %subview_17 = memref.subview %alloc_13[%28, 0] [%29, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    memref.copy %subview_14, %subview_16 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1]>>
    memref.copy %subview_15, %subview_17 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    %32 = bufferization.to_tensor %alloc_13 restrict writable : memref<4x1xi32> to tensor<4x1xi32>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg4 to offset: [%5], sizes: [%20, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [%21, %c1], strides: [%c1, %c0] : memref<*xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %alloc_20 = memref.alloc() : memref<4x1xi32>
    scf.if %27 {
      linalg.fill ins(%c0_i32 : i32) outs(%alloc_20 : memref<4x1xi32>)
    }
    %subview_21 = memref.subview %reinterpret_cast_18[0, 0] [%28, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_22 = memref.subview %reinterpret_cast_19[0, 0] [%29, 1] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[?, ?], offset: ?>>
    %subview_23 = memref.subview %alloc_20[0, 0] [%28, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1]>>
    %subview_24 = memref.subview %alloc_20[%28, 0] [%29, 1] [1, 1] : memref<4x1xi32> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    memref.copy %subview_21, %subview_23 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1]>>
    memref.copy %subview_22, %subview_24 : memref<?x1xi32, strided<[?, ?], offset: ?>> to memref<?x1xi32, strided<[1, 1], offset: ?>>
    %33 = bufferization.to_tensor %alloc_20 restrict writable : memref<4x1xi32> to tensor<4x1xi32>
    %34 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %1 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi sge, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %35 = linalg.fill ins(%arg6 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %36 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %35 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi slt, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %37 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%34, %36 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%34 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %38 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%31, %1 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi sge, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %39 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%37, %38 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%37 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %40 = linalg.fill ins(%arg7 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %41 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%31, %40 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi slt, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %42 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%39, %41 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%39 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %43 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%32, %1 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi sge, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %44 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%42, %43 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%42 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %45 = linalg.fill ins(%arg8 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %46 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%32, %45 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi slt, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %47 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%44, %46 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%44 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%33, %1 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi sge, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %49 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%47, %48 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%47 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %50 = linalg.fill ins(%arg9 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %51 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%33, %50 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%18 : tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: i32, %out: i1):
      %72 = arith.cmpi slt, %in, %in_26 : i32
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %52 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%49, %51 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%49 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %53 = arith.cmpi slt, %3, %c1_i64 : i64
    %54 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%52, %19 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%52 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %55 = linalg.fill ins(%53 : i1) outs(%18 : tensor<4x1xi1>) -> tensor<4x1xi1>
    %56 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%54, %55 : tensor<4x1xi1>, tensor<4x1xi1>) outs(%54 : tensor<4x1xi1>) {
    ^bb0(%in: i1, %in_26: i1, %out: i1):
      %72 = arith.andi %in, %in_26 : i1
      linalg.yield %72 : i1
    } -> tensor<4x1xi1>
    %57 = linalg.fill ins(%arg14 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %58 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %57 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%30 : tensor<4x1xi32>) {
    ^bb0(%in: i32, %in_26: i32, %out: i32):
      %72 = arith.muli %in, %in_26 : i32
      linalg.yield %72 : i32
    } -> tensor<4x1xi32>
    %59 = linalg.fill ins(%arg15 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %60 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%31, %59 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%31 : tensor<4x1xi32>) {
    ^bb0(%in: i32, %in_26: i32, %out: i32):
      %72 = arith.muli %in, %in_26 : i32
      linalg.yield %72 : i32
    } -> tensor<4x1xi32>
    %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%58, %60 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%58 : tensor<4x1xi32>) {
    ^bb0(%in: i32, %in_26: i32, %out: i32):
      %72 = arith.addi %in, %in_26 : i32
      linalg.yield %72 : i32
    } -> tensor<4x1xi32>
    %62 = linalg.fill ins(%arg16 : i32) outs(%0 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %63 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%32, %62 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%32 : tensor<4x1xi32>) {
    ^bb0(%in: i32, %in_26: i32, %out: i32):
      %72 = arith.muli %in, %in_26 : i32
      linalg.yield %72 : i32
    } -> tensor<4x1xi32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%61, %63 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%61 : tensor<4x1xi32>) {
    ^bb0(%in: i32, %in_26: i32, %out: i32):
      %72 = arith.addi %in, %in_26 : i32
      linalg.yield %72 : i32
    } -> tensor<4x1xi32>
    %65 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%64, %33 : tensor<4x1xi32>, tensor<4x1xi32>) outs(%64 : tensor<4x1xi32>) {
    ^bb0(%in: i32, %in_26: i32, %out: i32):
      %72 = arith.addi %in, %in_26 : i32
      linalg.yield %72 : i32
    } -> tensor<4x1xi32>
    %66 = arith.extsi %arg17 : i32 to i64
    %67 = linalg.fill ins(%66 : i64) outs(%8 : tensor<4x1xi64>) -> tensor<4x1xi64>
    %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %67 : tensor<4x1xi64>, tensor<4x1xi64>) outs(%15 : tensor<4x1xi64>) {
    ^bb0(%in: i64, %in_26: i64, %out: i64):
      %72 = arith.muli %in, %in_26 : i64
      linalg.yield %72 : i64
    } -> tensor<4x1xi64>
    %cast = memref.cast %arg5 : memref<*xf32> to memref<?xf32>
    %69 = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
    %70 = tensor.empty() : tensor<4x1xf32>
    %71 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%68, %56 : tensor<4x1xi64>, tensor<4x1xi1>) outs(%70 : tensor<4x1xf32>) {
    ^bb0(%in: i64, %in_26: i1, %out: f32):
      %72 = scf.if %in_26 -> (f32) {
        %73 = arith.index_cast %in : i64 to index
        %extracted = tensor.extract %69[%73] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        scf.yield %cst : f32
      }
      linalg.yield %72 : f32
    } -> tensor<4x1xf32>
    %cast_25 = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%65, %71, %56 : tensor<4x1xi32>, tensor<4x1xf32>, tensor<4x1xi1>) {
    ^bb0(%in: i32, %in_26: f32, %in_27: i1):
      scf.if %in_27 {
        %72 = arith.index_cast %in : i32 to index
        memref.store %in_26, %cast_25[%72] : memref<?xf32>
      }
      linalg.yield
    }
    return
  }
}


#map = affine_map<(d0) -> (d0)>
module {
  func.func @vstack_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = tptr.type_offset f32  : i64
    %c1_i32 = arith.constant 1 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4096 = arith.constant 4096 : index
    %cast = memref.cast %arg3 : memref<*xf32> to memref<1xf32>
    %1 = tptr.from_memref %cast : memref<1xf32> to <#tptr.default_memory_space>
    %cast_0 = memref.cast %arg2 : memref<*xf32> to memref<1xf32>
    %2 = tptr.from_memref %cast_0 : memref<1xf32> to <#tptr.default_memory_space>
    %cast_1 = memref.cast %arg1 : memref<*xf32> to memref<1xf32>
    %3 = tptr.from_memref %cast_1 : memref<1xf32> to <#tptr.default_memory_space>
    %cast_2 = memref.cast %arg0 : memref<*xf32> to memref<1xf32>
    %4 = tptr.from_memref %cast_2 : memref<1xf32> to <#tptr.default_memory_space>
    %5 = arith.extsi %arg17 : i32 to i64
    %6 = arith.extsi %arg18 : i32 to i64
    %7 = tensor.empty() : tensor<4096xi32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%7 : tensor<4096xi32>) {
    ^bb0(%out: i32):
      %48 = linalg.index 0 : index
      %49 = arith.index_cast %48 : index to i32
      linalg.yield %49 : i32
    } -> tensor<4096xi32>
    %9 = arith.cmpi eq, %6, %c0_i64 : i64
    %10 = arith.select %9, %4, %3 : !ptr.ptr<#tptr.default_memory_space>
    %11 = arith.cmpi eq, %6, %c2_i64 : i64
    %12 = arith.select %11, %2, %10 : !ptr.ptr<#tptr.default_memory_space>
    %13 = arith.cmpi eq, %6, %c3_i64 : i64
    %14 = arith.select %13, %1, %12 : !ptr.ptr<#tptr.default_memory_space>
    %15 = arith.select %9, %arg7, %arg8 : i32
    %16 = arith.select %11, %arg9, %15 : i32
    %17 = arith.select %13, %arg10, %16 : i32
    %18 = arith.select %9, %arg5, %arg6 : i32
    %19 = arith.select %11, %c1_i32, %18 : i32
    %20 = arith.select %13, %c1_i32, %19 : i32
    %21 = arith.extsi %arg12 : i32 to i64
    %22 = arith.extsi %20 : i32 to i64
    %23 = arith.muli %22, %21 : i64
    %24 = arith.muli %5, %c4096_i64 : i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = tensor.empty() : tensor<4096xi64>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%8 : tensor<4096xi32>) outs(%26 : tensor<4096xi64>) {
    ^bb0(%in: i32, %out: i64):
      %48 = arith.extsi %in : i32 to i64
      linalg.yield %48 : i64
    } -> tensor<4096xi64>
    %28 = linalg.fill ins(%24 : i64) outs(%26 : tensor<4096xi64>) -> tensor<4096xi64>
    %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%28, %27 : tensor<4096xi64>, tensor<4096xi64>) outs(%28 : tensor<4096xi64>) {
    ^bb0(%in: i64, %in_3: i64, %out: i64):
      %48 = arith.addi %in, %in_3 : i64
      linalg.yield %48 : i64
    } -> tensor<4096xi64>
    %30 = linalg.fill ins(%23 : i64) outs(%26 : tensor<4096xi64>) -> tensor<4096xi64>
    %31 = tensor.empty() : tensor<4096xi1>
    %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%29, %30 : tensor<4096xi64>, tensor<4096xi64>) outs(%31 : tensor<4096xi1>) {
    ^bb0(%in: i64, %in_3: i64, %out: i1):
      %48 = arith.cmpi slt, %in, %in_3 : i64
      linalg.yield %48 : i1
    } -> tensor<4096xi1>
    %33 = tensor.empty() : tensor<4096x!ptr.ptr<#tptr.default_memory_space>>
    %34 = linalg.fill ins(%14 : !ptr.ptr<#tptr.default_memory_space>) outs(%33 : tensor<4096x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<4096x!ptr.ptr<#tptr.default_memory_space>>
    %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%34, %29 : tensor<4096x!ptr.ptr<#tptr.default_memory_space>>, tensor<4096xi64>) outs(%34 : tensor<4096x!ptr.ptr<#tptr.default_memory_space>>) {
    ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_3: i64, %out: !ptr.ptr<#tptr.default_memory_space>):
      %48 = arith.muli %in_3, %0 : i64
      %49 = tptr.ptradd %in %48 : <#tptr.default_memory_space>, i64 to <#tptr.default_memory_space>
      linalg.yield %49 : !ptr.ptr<#tptr.default_memory_space>
    } -> tensor<4096x!ptr.ptr<#tptr.default_memory_space>>
    %36 = arith.addi %arg11, %17 : i32
    %37 = arith.extsi %36 : i32 to i64
    %38 = arith.muli %37, %21 : i64
    %39 = arith.index_cast %38 : i64 to index
    %40 = arith.addi %39, %25 : index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %41 = tensor.empty() : tensor<4096xf32>
    %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%35, %32 : tensor<4096x!ptr.ptr<#tptr.default_memory_space>>, tensor<4096xi1>) outs(%41 : tensor<4096xf32>) {
    ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_3: i1, %out: f32):
      %48 = tptr.to_memref %in : <#tptr.default_memory_space> to memref<1xf32>
      %49 = scf.if %in_3 -> (f32) {
        %50 = memref.load %48[%c0] : memref<1xf32>
        scf.yield %50 : f32
      } else {
        scf.yield %cst : f32
      }
      linalg.yield %49 : f32
    } -> tensor<4096xf32>
    %43 = arith.addi %25, %c4096 : index
    %44 = arith.index_cast %23 : i64 to index
    %45 = arith.minsi %43, %44 : index
    %46 = arith.maxsi %45, %25 : index
    %47 = arith.subi %46, %25 : index
    %extracted_slice = tensor.extract_slice %42[0] [%47] [1] : tensor<4096xf32> to tensor<?xf32>
    %subview = memref.subview %reinterpret_cast[0] [%47] [1] : memref<4096xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


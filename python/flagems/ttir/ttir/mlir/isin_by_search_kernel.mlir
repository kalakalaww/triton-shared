#map = affine_map<(d0) -> (d0)>
module {
  func.func @isin_by_search_kernel(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: memref<*xi1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %0 = tptr.type_offset i1  : i64
    %c0 = arith.constant 0 : index
    %1 = tptr.type_offset i32  : i32
    %false = arith.constant false
    %c512 = arith.constant 512 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c512_i64 = arith.constant 512 : i64
    %cast = memref.cast %arg2 : memref<*xi1> to memref<1xi1>
    %2 = tptr.from_memref %cast : memref<1xi1> to <#tptr.default_memory_space>
    %cast_0 = memref.cast %arg1 : memref<*xi32> to memref<1xi32>
    %3 = tptr.from_memref %cast_0 : memref<1xi32> to <#tptr.default_memory_space>
    %4 = tensor.empty() : tensor<512xi1>
    %5 = linalg.fill ins(%false : i1) outs(%4 : tensor<512xi1>) -> tensor<512xi1>
    %6 = tensor.empty() : tensor<512xi32>
    %7 = linalg.fill ins(%c1_i32 : i32) outs(%6 : tensor<512xi32>) -> tensor<512xi32>
    %8 = linalg.fill ins(%c2_i32 : i32) outs(%6 : tensor<512xi32>) -> tensor<512xi32>
    %9 = linalg.fill ins(%c0_i32 : i32) outs(%6 : tensor<512xi32>) -> tensor<512xi32>
    %10 = arith.extsi %arg8 : i32 to i64
    %11 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<512xi32>) {
    ^bb0(%out: i32):
      %37 = linalg.index 0 : index
      %38 = arith.index_cast %37 : index to i32
      linalg.yield %38 : i32
    } -> tensor<512xi32>
    %12 = arith.muli %10, %c512_i64 : i64
    %13 = arith.index_cast %12 : i64 to index
    %14 = tensor.empty() : tensor<512xi64>
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%11 : tensor<512xi32>) outs(%14 : tensor<512xi64>) {
    ^bb0(%in: i32, %out: i64):
      %37 = arith.extsi %in : i32 to i64
      linalg.yield %37 : i64
    } -> tensor<512xi64>
    %16 = linalg.fill ins(%12 : i64) outs(%14 : tensor<512xi64>) -> tensor<512xi64>
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%16, %15 : tensor<512xi64>, tensor<512xi64>) outs(%16 : tensor<512xi64>) {
    ^bb0(%in: i64, %in_2: i64, %out: i64):
      %37 = arith.addi %in, %in_2 : i64
      linalg.yield %37 : i64
    } -> tensor<512xi64>
    %18 = arith.extsi %arg3 : i32 to i64
    %19 = linalg.fill ins(%18 : i64) outs(%14 : tensor<512xi64>) -> tensor<512xi64>
    %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%17, %19 : tensor<512xi64>, tensor<512xi64>) outs(%4 : tensor<512xi1>) {
    ^bb0(%in: i64, %in_2: i64, %out: i1):
      %37 = arith.cmpi slt, %in, %in_2 : i64
      linalg.yield %37 : i1
    } -> tensor<512xi1>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [512], strides: [1] : memref<*xi32> to memref<512xi32, strided<[1], offset: ?>>
    %21 = arith.addi %13, %c512 : index
    %22 = arith.index_cast %arg3 : i32 to index
    %23 = arith.minsi %21, %22 : index
    %24 = arith.maxsi %23, %13 : index
    %25 = arith.subi %24, %13 : index
    %alloc = memref.alloc() : memref<512xi32>
    %subview = memref.subview %reinterpret_cast[0] [%25] [1] : memref<512xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%25] [1] : memref<512xi32> to memref<?xi32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
    %26 = bufferization.to_tensor %alloc restrict writable : memref<512xi32> to tensor<512xi32>
    %27 = linalg.fill ins(%arg4 : i32) outs(%6 : tensor<512xi32>) -> tensor<512xi32>
    %28 = arith.cmpi sgt, %arg4, %c0_i32 : i32
    %29 = linalg.fill ins(%28 : i1) outs(%4 : tensor<512xi1>) -> tensor<512xi1>
    %30 = tensor.empty() : tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %31 = linalg.fill ins(%3 : !ptr.ptr<#tptr.default_memory_space>) outs(%30 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %32:4 = scf.for %arg11 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg12 = %5, %arg13 = %9, %arg14 = %27, %arg15 = %29) -> (tensor<512xi1>, tensor<512xi32>, tensor<512xi32>, tensor<512xi1>)  : i32 {
      %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg14, %arg13 : tensor<512xi32>, tensor<512xi32>) outs(%arg14 : tensor<512xi32>) {
      ^bb0(%in: i32, %in_2: i32, %out: i32):
        %54 = arith.subi %in, %in_2 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%37, %8 : tensor<512xi32>, tensor<512xi32>) outs(%37 : tensor<512xi32>) {
      ^bb0(%in: i32, %in_2: i32, %out: i32):
        %54 = arith.divsi %in, %in_2 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %38 : tensor<512xi32>, tensor<512xi32>) outs(%arg13 : tensor<512xi32>) {
      ^bb0(%in: i32, %in_2: i32, %out: i32):
        %54 = arith.addi %in, %in_2 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %40 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg15, %39, %9 : tensor<512xi1>, tensor<512xi32>, tensor<512xi32>) outs(%39 : tensor<512xi32>) {
      ^bb0(%in: i1, %in_2: i32, %in_3: i32, %out: i32):
        %54 = arith.select %in, %in_2, %in_3 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%31, %40 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>, tensor<512xi32>) outs(%31 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>) {
      ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_2: i32, %out: !ptr.ptr<#tptr.default_memory_space>):
        %54 = arith.muli %in_2, %1 : i32
        %55 = tptr.ptradd %in %54 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
        linalg.yield %55 : !ptr.ptr<#tptr.default_memory_space>
      } -> tensor<512x!ptr.ptr<#tptr.default_memory_space>>
      %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%41, %arg15 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>, tensor<512xi1>) outs(%6 : tensor<512xi32>) {
      ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_2: i1, %out: i32):
        %54 = tptr.to_memref %in : <#tptr.default_memory_space> to memref<1xi32>
        %55 = scf.if %in_2 -> (i32) {
          %56 = memref.load %54[%c0] : memref<1xi32>
          scf.yield %56 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        linalg.yield %55 : i32
      } -> tensor<512xi32>
      %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%42, %26 : tensor<512xi32>, tensor<512xi32>) outs(%4 : tensor<512xi1>) {
      ^bb0(%in: i32, %in_2: i32, %out: i1):
        %54 = arith.cmpi eq, %in, %in_2 : i32
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %44 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg12, %43 : tensor<512xi1>, tensor<512xi1>) outs(%arg12 : tensor<512xi1>) {
      ^bb0(%in: i1, %in_2: i1, %out: i1):
        %54 = arith.ori %in, %in_2 : i1
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %45 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg15, %44, %arg12 : tensor<512xi1>, tensor<512xi1>, tensor<512xi1>) outs(%arg15 : tensor<512xi1>) {
      ^bb0(%in: i1, %in_2: i1, %in_3: i1, %out: i1):
        %54 = arith.select %in, %in_2, %in_3 : i1
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %46 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%42, %26 : tensor<512xi32>, tensor<512xi32>) outs(%4 : tensor<512xi1>) {
      ^bb0(%in: i32, %in_2: i32, %out: i1):
        %54 = arith.cmpi slt, %in, %in_2 : i32
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg15, %46 : tensor<512xi1>, tensor<512xi1>) outs(%arg15 : tensor<512xi1>) {
      ^bb0(%in: i1, %in_2: i1, %out: i1):
        %54 = arith.andi %in, %in_2 : i1
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %48 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%40, %7 : tensor<512xi32>, tensor<512xi32>) outs(%40 : tensor<512xi32>) {
      ^bb0(%in: i32, %in_2: i32, %out: i32):
        %54 = arith.addi %in, %in_2 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %49 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%47, %48, %arg13 : tensor<512xi1>, tensor<512xi32>, tensor<512xi32>) outs(%48 : tensor<512xi32>) {
      ^bb0(%in: i1, %in_2: i32, %in_3: i32, %out: i32):
        %54 = arith.select %in, %in_2, %in_3 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%42, %26 : tensor<512xi32>, tensor<512xi32>) outs(%4 : tensor<512xi1>) {
      ^bb0(%in: i32, %in_2: i32, %out: i1):
        %54 = arith.cmpi sgt, %in, %in_2 : i32
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %51 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg15, %50 : tensor<512xi1>, tensor<512xi1>) outs(%arg15 : tensor<512xi1>) {
      ^bb0(%in: i1, %in_2: i1, %out: i1):
        %54 = arith.andi %in, %in_2 : i1
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      %52 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%51, %40, %arg14 : tensor<512xi1>, tensor<512xi32>, tensor<512xi32>) outs(%40 : tensor<512xi32>) {
      ^bb0(%in: i1, %in_2: i32, %in_3: i32, %out: i32):
        %54 = arith.select %in, %in_2, %in_3 : i32
        linalg.yield %54 : i32
      } -> tensor<512xi32>
      %53 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%49, %52 : tensor<512xi32>, tensor<512xi32>) outs(%4 : tensor<512xi1>) {
      ^bb0(%in: i32, %in_2: i32, %out: i1):
        %54 = arith.cmpi slt, %in, %in_2 : i32
        linalg.yield %54 : i1
      } -> tensor<512xi1>
      scf.yield %45, %49, %52, %53 : tensor<512xi1>, tensor<512xi32>, tensor<512xi32>, tensor<512xi1>
    }
    %33 = linalg.fill ins(%2 : !ptr.ptr<#tptr.default_memory_space>) outs(%30 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%33, %17 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>, tensor<512xi64>) outs(%33 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>) {
    ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_2: i64, %out: !ptr.ptr<#tptr.default_memory_space>):
      %37 = arith.muli %in_2, %0 : i64
      %38 = tptr.ptradd %in %37 : <#tptr.default_memory_space>, i64 to <#tptr.default_memory_space>
      linalg.yield %38 : !ptr.ptr<#tptr.default_memory_space>
    } -> tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %35 = tensor.empty() : tensor<512xi8>
    %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%32#0 : tensor<512xi1>) outs(%35 : tensor<512xi8>) {
    ^bb0(%in: i1, %out: i8):
      %37 = arith.extui %in : i1 to i8
      linalg.yield %37 : i8
    } -> tensor<512xi8>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%34, %36, %20 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>, tensor<512xi8>, tensor<512xi1>) {
    ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_2: i8, %in_3: i1):
      scf.if %in_3 {
        %37 = tptr.to_memref %in : <#tptr.default_memory_space> to memref<1xi8>
        memref.store %in_2, %37[%c0] : memref<1xi8>
      }
      linalg.yield
    }
    return
  }
}


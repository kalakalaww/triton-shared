#map = affine_map<(d0) -> (d0)>
module {
  func.func @all_kernel_2(%arg0: memref<*xi1> {tt.divisibility = 16 : i32}, %arg1: memref<*xi1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c0 = arith.constant 0 : index
    %0 = tptr.type_offset i1  : i32
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %cast = memref.cast %arg1 : memref<*xi1> to memref<1xi1>
    %1 = tptr.from_memref %cast : memref<1xi1> to <#tptr.default_memory_space>
    %cast_0 = memref.cast %arg0 : memref<*xi1> to memref<1xi1>
    %2 = tptr.from_memref %cast_0 : memref<1xi1> to <#tptr.default_memory_space>
    %3 = tensor.empty() : tensor<512xi8>
    %4 = linalg.fill ins(%c1_i8 : i8) outs(%3 : tensor<512xi8>) -> tensor<512xi8>
    %5 = linalg.fill ins(%c0_i8 : i8) outs(%3 : tensor<512xi8>) -> tensor<512xi8>
    %6 = tensor.empty() : tensor<512xi32>
    %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<512xi32>) {
    ^bb0(%out: i32):
      %19 = linalg.index 0 : index
      %20 = arith.index_cast %19 : index to i32
      linalg.yield %20 : i32
    } -> tensor<512xi32>
    %8 = tensor.empty() : tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %9 = linalg.fill ins(%2 : !ptr.ptr<#tptr.default_memory_space>) outs(%8 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %7 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>, tensor<512xi32>) outs(%9 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>) {
    ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_1: i32, %out: !ptr.ptr<#tptr.default_memory_space>):
      %19 = arith.muli %in_1, %0 : i32
      %20 = tptr.ptradd %in %19 : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
      linalg.yield %20 : !ptr.ptr<#tptr.default_memory_space>
    } -> tensor<512x!ptr.ptr<#tptr.default_memory_space>>
    %11 = linalg.fill ins(%arg2 : i32) outs(%6 : tensor<512xi32>) -> tensor<512xi32>
    %12 = tensor.empty() : tensor<512xi1>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %11 : tensor<512xi32>, tensor<512xi32>) outs(%12 : tensor<512xi1>) {
    ^bb0(%in: i32, %in_1: i32, %out: i1):
      %19 = arith.cmpi slt, %in, %in_1 : i32
      linalg.yield %19 : i1
    } -> tensor<512xi1>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%10, %13, %4 : tensor<512x!ptr.ptr<#tptr.default_memory_space>>, tensor<512xi1>, tensor<512xi8>) outs(%3 : tensor<512xi8>) {
    ^bb0(%in: !ptr.ptr<#tptr.default_memory_space>, %in_1: i1, %in_2: i8, %out: i8):
      %19 = tptr.to_memref %in : <#tptr.default_memory_space> to memref<1xi8>
      %20 = scf.if %in_1 -> (i8) {
        %21 = memref.load %19[%c0] : memref<1xi8>
        scf.yield %21 : i8
      } else {
        scf.yield %in_2 : i8
      }
      linalg.yield %20 : i8
    } -> tensor<512xi8>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%14, %5 : tensor<512xi8>, tensor<512xi8>) outs(%12 : tensor<512xi1>) {
    ^bb0(%in: i8, %in_1: i8, %out: i1):
      %19 = arith.cmpi ne, %in, %in_1 : i8
      linalg.yield %19 : i1
    } -> tensor<512xi1>
    %16 = bufferization.alloc_tensor() : tensor<i1>
    %inserted = tensor.insert %true into %16[] : tensor<i1>
    %reduced = linalg.reduce ins(%15 : tensor<512xi1>) outs(%inserted : tensor<i1>) dimensions = [0] 
      (%in: i1, %init: i1) {
        %19 = arith.andi %in, %init : i1
        linalg.yield %19 : i1
      }
    %extracted = tensor.extract %reduced[] : tensor<i1>
    %17 = arith.extui %extracted : i1 to i8
    %18 = tptr.to_memref %1 : <#tptr.default_memory_space> to memref<1xi8>
    memref.store %17, %18[%c0] : memref<1xi8>
    return
  }
}


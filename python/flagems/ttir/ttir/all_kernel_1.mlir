#map = affine_map<(d0) -> (d0)>
module {
  func.func @all_kernel_1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0 = arith.constant 0 : index
    %0 = tptr.type_offset i1  : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1024_i64 = arith.constant 1024 : i64
    %true = arith.constant true
    %cast = memref.cast %arg1 : memref<*xi1> to memref<1xi1>
    %1 = tptr.from_memref %cast : memref<1xi1> to <#tptr.default_memory_space>
    %2 = tensor.empty() : tensor<1024xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
    %4 = arith.extsi %arg7 : i32 to i64
    %5 = arith.muli %4, %c1024_i64 : i64
    %6 = arith.index_cast %5 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %7 = arith.addi %6, %c1024 : index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.minsi %7, %8 : index
    %10 = arith.maxsi %9, %6 : index
    %11 = arith.subi %10, %6 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %12 = arith.cmpi slt, %11, %c1024 : index
    scf.if %12 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<1024xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%11] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%11] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %13 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %14 = tensor.empty() : tensor<1024xi1>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%13, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%14 : tensor<1024xi1>) {
    ^bb0(%in: f32, %in_2: f32, %out: i1):
      %21 = arith.cmpf une, %in, %in_2 : f32
      linalg.yield %21 : i1
    } -> tensor<1024xi1>
    %16 = bufferization.alloc_tensor() : tensor<i1>
    %inserted = tensor.insert %true into %16[] : tensor<i1>
    %reduced = linalg.reduce ins(%15 : tensor<1024xi1>) outs(%inserted : tensor<i1>) dimensions = [0] 
      (%in: i1, %init: i1) {
        %21 = arith.andi %in, %init : i1
        linalg.yield %21 : i1
      }
    %extracted = tensor.extract %reduced[] : tensor<i1>
    %17 = arith.muli %4, %0 : i64
    %18 = tptr.ptradd %1 %17 : <#tptr.default_memory_space>, i64 to <#tptr.default_memory_space>
    %19 = arith.extui %extracted : i1 to i8
    %20 = tptr.to_memref %18 : <#tptr.default_memory_space> to memref<1xi8>
    memref.store %19, %20[%c0] : memref<1xi8>
    return
  }
}


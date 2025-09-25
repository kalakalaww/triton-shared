#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %arg12, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %arg12, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %arg3 : i32 to index
    %20 = arith.index_cast %arg6 : i32 to index
    %21 = arith.muli %16, %20 : index
    %22 = arith.muli %19, %20 : index
    %23 = arith.index_cast %arg7 : i32 to index
    %24 = arith.index_cast %arg4 : i32 to index
    %25 = arith.addi %arg5, %c31_i32 : i32
    %26 = arith.divsi %25, %c32_i32 : i32
    %27 = arith.muli %arg7, %c32_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29:3 = scf.for %arg15 = %c0_i32 to %26 step %c1_i32 iter_args(%arg16 = %1, %arg17 = %21, %arg18 = %c0) -> (tensor<128x128xf32>, index, index)  : i32 {
      %43 = arith.addi %arg18, %18 : index
      %44 = arith.remsi %43, %24 : index
      %45 = arith.subi %43, %44 : index
      %46 = arith.addi %44, %c128 : index
      %47 = arith.minsi %46, %24 : index
      %48 = arith.subi %47, %44 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%c32, %48], strides: [%23, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %49 = arith.subi %c128, %48 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [%c32, %49], strides: [%23, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %50 = arith.remsi %arg17, %20 : index
      %51 = arith.addi %22, %50 : index
      %52 = arith.subi %51, %arg17 : index
      %53 = arith.divsi %52, %20 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%arg17], sizes: [%53, %c32], strides: [%20, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %54 = arith.subi %c128, %53 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [%54, %c32], strides: [%20, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %55 = arith.muli %arg15, %c32_i32 : i32
      %56 = arith.subi %arg5, %55 : i32
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.minsi %57, %c32 : index
      %59 = arith.maxsi %58, %c0 : index
      %alloc = memref.alloc() : memref<128x32xf32>
      %60 = arith.cmpi slt, %59, %c32 : index
      scf.if %60 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<128x32xf32>)
      }
      %61 = arith.minsi %53, %c128 : index
      %62 = arith.subi %c128, %61 : index
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%61, %59] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%62, %59] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_6 = memref.subview %alloc[0, 0] [%61, %59] [1, 1] : memref<128x32xf32> to memref<?x?xf32, strided<[32, 1]>>
      %subview_7 = memref.subview %alloc[%61, 0] [%62, %59] [1, 1] : memref<128x32xf32> to memref<?x?xf32, strided<[32, 1], offset: ?>>
      memref.copy %subview_4, %subview_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[32, 1]>>
      memref.copy %subview_5, %subview_7 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
      %63 = bufferization.to_tensor %alloc restrict writable : memref<128x32xf32> to tensor<128x32xf32>
      %alloc_8 = memref.alloc() : memref<32x128xf32>
      scf.if %60 {
        linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<32x128xf32>)
      }
      %64 = arith.minsi %48, %c128 : index
      %65 = arith.subi %c128, %64 : index
      %subview_9 = memref.subview %reinterpret_cast_0[0, 0] [%59, %64] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_10 = memref.subview %reinterpret_cast_1[0, 0] [%59, %65] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %alloc_8[0, 0] [%59, %64] [1, 1] : memref<32x128xf32> to memref<?x?xf32, strided<[128, 1]>>
      %subview_12 = memref.subview %alloc_8[0, %64] [%59, %65] [1, 1] : memref<32x128xf32> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      memref.copy %subview_9, %subview_11 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[128, 1]>>
      memref.copy %subview_10, %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      %66 = bufferization.to_tensor %alloc_8 restrict writable : memref<32x128xf32> to tensor<32x128xf32>
      %67 = linalg.matmul ins(%63, %66 : tensor<128x32xf32>, tensor<32x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %68 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg16, %67 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg16 : tensor<128x128xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %71 = arith.addf %in, %in_13 : f32
        linalg.yield %71 : f32
      } -> tensor<128x128xf32>
      %69 = arith.addi %arg17, %c32 : index
      %70 = arith.addi %arg18, %28 : index
      scf.yield %68, %69, %70 : tensor<128x128xf32>, index, index
    }
    %30 = arith.index_cast %arg8 : i32 to index
    %31 = arith.muli %16, %30 : index
    %32 = arith.addi %31, %18 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [128, 128], strides: [%30, 1] : memref<*xf32> to memref<128x128xf32, strided<[?, 1], offset: ?>>
    %33 = arith.addi %16, %c128 : index
    %34 = arith.minsi %33, %19 : index
    %35 = arith.maxsi %34, %16 : index
    %36 = arith.subi %35, %16 : index
    %37 = arith.addi %18, %c128 : index
    %38 = arith.minsi %37, %24 : index
    %39 = arith.maxsi %38, %18 : index
    %40 = arith.subi %39, %18 : index
    %41 = arith.minsi %36, %c128 : index
    %42 = arith.minsi %40, %c128 : index
    %extracted_slice = tensor.extract_slice %29#0[0, 0] [%41, %42] [1, 1] : tensor<128x128xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%41, %42] [1, 1] : memref<128x128xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}


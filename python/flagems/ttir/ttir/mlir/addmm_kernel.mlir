#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @addmm_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: memref<*xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c256 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = arith.extsi %arg16 : i32 to i64
    %3 = arith.extsi %arg17 : i32 to i64
    %4 = arith.muli %2, %c128_i64 : i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = arith.muli %3, %c256_i64 : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.index_cast %arg9 : i32 to index
    %9 = arith.muli %5, %8 : index
    %10 = arith.index_cast %arg10 : i32 to index
    %11 = arith.addi %arg8, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %arg10, %c64_i32 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15:3 = scf.for %arg19 = %c0_i32 to %12 step %c1_i32 iter_args(%arg20 = %1, %arg21 = %9, %arg22 = %c0) -> (tensor<128x256xf32>, index, index)  : i32 {
      %45 = arith.addi %arg22, %7 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [64, 256], strides: [%10, 1] : memref<*xf32> to memref<64x256xf32, strided<[?, 1], offset: ?>>
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%arg21], sizes: [128, 64], strides: [%8, 1] : memref<*xf32> to memref<128x64xf32, strided<[?, 1], offset: ?>>
      %46 = arith.muli %arg19, %c64_i32 : i32
      %47 = arith.subi %arg8, %46 : i32
      %48 = arith.addi %5, %c128 : index
      %49 = arith.index_cast %arg6 : i32 to index
      %50 = arith.minsi %48, %49 : index
      %51 = arith.maxsi %50, %5 : index
      %52 = arith.subi %51, %5 : index
      %53 = arith.index_cast %47 : i32 to index
      %54 = arith.minsi %53, %c64 : index
      %55 = arith.maxsi %54, %c0 : index
      %56 = arith.minsi %52, %c128 : index
      %57 = arith.minsi %55, %c64 : index
      %alloc_5 = memref.alloc() : memref<128x64xf32>
      %58 = arith.cmpi slt, %56, %c128 : index
      %59 = arith.cmpi slt, %57, %c64 : index
      %60 = arith.ori %58, %59 : i1
      scf.if %60 {
        linalg.fill ins(%cst : f32) outs(%alloc_5 : memref<128x64xf32>)
      }
      %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%56, %57] [1, 1] : memref<128x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%56, %57] [1, 1] : memref<128x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %61 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x64xf32> to tensor<128x64xf32>
      %62 = arith.addi %7, %c256 : index
      %63 = arith.index_cast %arg7 : i32 to index
      %64 = arith.minsi %62, %63 : index
      %65 = arith.maxsi %64, %7 : index
      %66 = arith.subi %65, %7 : index
      %67 = arith.minsi %66, %c256 : index
      %alloc_8 = memref.alloc() : memref<64x256xf32>
      %68 = arith.cmpi slt, %67, %c256 : index
      %69 = arith.ori %59, %68 : i1
      scf.if %69 {
        linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<64x256xf32>)
      }
      %subview_9 = memref.subview %reinterpret_cast_3[0, 0] [%57, %67] [1, 1] : memref<64x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[0, 0] [%57, %67] [1, 1] : memref<64x256xf32> to memref<?x?xf32, strided<[256, 1]>>
      memref.copy %subview_9, %subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
      %70 = bufferization.to_tensor %alloc_8 restrict writable : memref<64x256xf32> to tensor<64x256xf32>
      %71 = linalg.matmul ins(%61, %70 : tensor<128x64xf32>, tensor<64x256xf32>) outs(%1 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %72 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg20, %71 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%arg20 : tensor<128x256xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %75 = arith.addf %in, %in_11 : f32
        linalg.yield %75 : f32
      } -> tensor<128x256xf32>
      %73 = arith.addi %arg21, %c64 : index
      %74 = arith.addi %arg22, %14 : index
      scf.yield %72, %73, %74 : tensor<128x256xf32>, index, index
    }
    %16 = arith.index_cast %arg12 : i32 to index
    %17 = arith.muli %5, %16 : index
    %18 = arith.addi %17, %7 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [128, 256], strides: [%16, 1] : memref<*xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
    %19 = arith.index_cast %arg11 : i32 to index
    %20 = arith.muli %5, %19 : index
    %21 = arith.addi %20, %7 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [128, 256], strides: [%19, 1] : memref<*xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
    %22 = arith.addi %5, %c128 : index
    %23 = arith.index_cast %arg6 : i32 to index
    %24 = arith.minsi %22, %23 : index
    %25 = arith.maxsi %24, %5 : index
    %26 = arith.subi %25, %5 : index
    %27 = arith.addi %7, %c256 : index
    %28 = arith.index_cast %arg7 : i32 to index
    %29 = arith.minsi %27, %28 : index
    %30 = arith.maxsi %29, %7 : index
    %31 = arith.subi %30, %7 : index
    %32 = arith.minsi %26, %c128 : index
    %33 = arith.minsi %31, %c256 : index
    %alloc = memref.alloc() : memref<128x256xf32>
    %34 = arith.cmpi slt, %32, %c128 : index
    %35 = arith.cmpi slt, %33, %c256 : index
    %36 = arith.ori %34, %35 : i1
    scf.if %36 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<128x256xf32>)
    }
    %subview = memref.subview %reinterpret_cast_0[0, 0] [%32, %33] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0] [%32, %33] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
    %37 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf32> to tensor<128x256xf32>
    %38 = arith.sitofp %arg4 : i32 to f32
    %39 = linalg.fill ins(%38 : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%15#0, %39 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%15#0 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %45 = arith.mulf %in, %in_3 : f32
      linalg.yield %45 : f32
    } -> tensor<128x256xf32>
    %41 = arith.sitofp %arg5 : i32 to f32
    %42 = linalg.fill ins(%41 : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%37, %42 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%37 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %45 = arith.mulf %in, %in_3 : f32
      linalg.yield %45 : f32
    } -> tensor<128x256xf32>
    %44 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%40, %43 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%40 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %45 = arith.addf %in, %in_3 : f32
      linalg.yield %45 : f32
    } -> tensor<128x256xf32>
    %extracted_slice = tensor.extract_slice %44[0, 0] [%32, %33] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
    %subview_2 = memref.subview %reinterpret_cast[0, 0] [%32, %33] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}


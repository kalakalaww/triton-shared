#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @bmm_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = arith.extsi %arg11 : i32 to i64
    %3 = arith.extsi %arg3 : i32 to i64
    %4 = arith.muli %2, %3 : i64
    %5 = arith.extsi %arg5 : i32 to i64
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.muli %4, %5 : i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = arith.muli %2, %5 : i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.index_cast %arg4 : i32 to index
    %12 = arith.muli %9, %10 : i64
    %13 = arith.index_cast %12 : i64 to index
    %14 = arith.muli %4, %10 : i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = arith.extsi %arg9 : i32 to i64
    %17 = arith.extsi %arg10 : i32 to i64
    %18 = arith.muli %16, %c32_i64 : i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.muli %17, %c32_i64 : i64
    %21 = arith.index_cast %20 : i64 to index
    %22 = arith.muli %19, %6 : index
    %23 = arith.addi %8, %22 : index
    %24 = arith.muli %19, %11 : index
    %25 = arith.addi %15, %24 : index
    %26 = arith.addi %25, %21 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [32, 32], strides: [%11, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
    %27 = arith.addi %arg5, %c31_i32 : i32
    %28 = arith.divsi %27, %c32_i32 : i32
    %29 = arith.muli %arg4, %c32_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31:3 = scf.for %arg12 = %c0_i32 to %28 step %c1_i32 iter_args(%arg13 = %23, %arg14 = %13, %arg15 = %1) -> (index, index, tensor<32x32xf32>)  : i32 {
      %36 = arith.addi %arg14, %21 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [32, 32], strides: [%11, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%arg13], sizes: [32, 32], strides: [%6, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x32xf32>
      memref.copy %reinterpret_cast_1, %alloc : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<32x32xf32>
      %37 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
      %38 = arith.addi %21, %c32 : index
      %39 = arith.minsi %38, %11 : index
      %40 = arith.maxsi %39, %21 : index
      %41 = arith.subi %40, %21 : index
      %alloc_2 = memref.alloc() : memref<32x32xf32>
      %subview_3 = memref.subview %reinterpret_cast_0[0, 0] [32, %41] [1, 1] : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<32x?xf32, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %alloc_2[0, 0] [32, %41] [1, 1] : memref<32x32xf32> to memref<32x?xf32, strided<[32, 1]>>
      memref.copy %subview_3, %subview_4 : memref<32x?xf32, strided<[?, 1], offset: ?>> to memref<32x?xf32, strided<[32, 1]>>
      %42 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
      %43 = arith.addi %arg13, %c32 : index
      %44 = arith.addi %arg14, %30 : index
      %45 = linalg.matmul ins(%37, %42 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %46 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %45 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg15 : tensor<32x32xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %47 = arith.addf %in, %in_5 : f32
        linalg.yield %47 : f32
      } -> tensor<32x32xf32>
      scf.yield %43, %44, %46 : index, index, tensor<32x32xf32>
    }
    %32 = arith.addi %21, %c32 : index
    %33 = arith.minsi %32, %11 : index
    %34 = arith.maxsi %33, %21 : index
    %35 = arith.subi %34, %21 : index
    %extracted_slice = tensor.extract_slice %31#2[0, 0] [32, %35] [1, 1] : tensor<32x32xf32> to tensor<32x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [32, %35] [1, 1] : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<32x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<32x?xf32>, memref<32x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}


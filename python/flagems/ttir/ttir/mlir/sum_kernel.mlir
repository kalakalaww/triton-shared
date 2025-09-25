#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @sum_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i64 = arith.constant 8 : i64
    %0 = tensor.empty() : tensor<8x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %2 = arith.extsi %arg7 : i32 to i64
    %3 = arith.muli %2, %c8_i64 : i64
    %4 = arith.index_cast %3 : i64 to index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = arith.muli %4, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [8, 1], strides: [1, 1] : memref<*xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
    %7 = scf.for %arg10 = %c0_i32 to %arg3 step %c1024_i32 iter_args(%arg11 = %1) -> (tensor<8x1024xf32>)  : i32 {
      %16 = arith.index_cast %arg10 : i32 to index
      %17 = arith.addi %6, %16 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [8, 1024], strides: [%5, 1] : memref<*xf32> to memref<8x1024xf32, strided<[?, 1], offset: ?>>
      %18 = arith.addi %4, %c8 : index
      %19 = arith.index_cast %arg2 : i32 to index
      %20 = arith.minsi %18, %19 : index
      %21 = arith.maxsi %20, %4 : index
      %22 = arith.subi %21, %4 : index
      %23 = arith.addi %16, %c1024 : index
      %24 = arith.minsi %23, %5 : index
      %25 = arith.maxsi %24, %16 : index
      %26 = arith.subi %25, %16 : index
      %27 = arith.minsi %22, %c8 : index
      %28 = arith.minsi %26, %c1024 : index
      %alloc = memref.alloc() : memref<8x1024xf32>
      %29 = arith.cmpi slt, %27, %c8 : index
      %30 = arith.cmpi slt, %28, %c1024 : index
      %31 = arith.ori %29, %30 : i1
      scf.if %31 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<8x1024xf32>)
      }
      %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [%27, %28] [1, 1] : memref<8x1024xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[0, 0] [%27, %28] [1, 1] : memref<8x1024xf32> to memref<?x?xf32, strided<[1024, 1]>>
      memref.copy %subview_1, %subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[1024, 1]>>
      %32 = bufferization.to_tensor %alloc restrict writable : memref<8x1024xf32> to tensor<8x1024xf32>
      %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg11, %32 : tensor<8x1024xf32>, tensor<8x1024xf32>) outs(%arg11 : tensor<8x1024xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %34 = arith.addf %in, %in_3 : f32
        linalg.yield %34 : f32
      } -> tensor<8x1024xf32>
      scf.yield %33 : tensor<8x1024xf32>
    }
    %8 = tensor.empty() : tensor<1024x8xf32>
    %transposed = linalg.transpose ins(%7 : tensor<8x1024xf32>) outs(%8 : tensor<1024x8xf32>) permutation = [1, 0] 
    %9 = tensor.empty() : tensor<8xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<8xf32>) -> tensor<8xf32>
    %reduced = linalg.reduce ins(%transposed : tensor<1024x8xf32>) outs(%10 : tensor<8xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %16 = arith.addf %in, %init : f32
        linalg.yield %16 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [8, 1] : tensor<8xf32> into tensor<8x1xf32>
    %11 = arith.addi %4, %c8 : index
    %12 = arith.index_cast %arg2 : i32 to index
    %13 = arith.minsi %11, %12 : index
    %14 = arith.maxsi %13, %4 : index
    %15 = arith.subi %14, %4 : index
    %extracted_slice = tensor.extract_slice %expanded[0, 0] [%15, 1] [1, 1] : tensor<8x1xf32> to tensor<?x1xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%15, 1] [1, 1] : memref<8x1xf32, strided<[1, 1], offset: ?>> to memref<?x1xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x1xf32>, memref<?x1xf32, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}


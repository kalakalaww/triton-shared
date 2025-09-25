#map = affine_map<(d0) -> (d0)>
module {
  func.func @softmax_kernel_inner(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = arith.extsi %arg7 : i32 to i64
    %1 = arith.extsi %arg3 : i32 to i64
    %2 = arith.muli %0, %1 : i64
    %3 = arith.index_cast %2 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1], offset: ?>>
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.minsi %4, %c512 : index
    %6 = arith.maxsi %5, %c0 : index
    %alloc = memref.alloc() : memref<512xf32>
    %7 = arith.cmpi slt, %6, %c512 : index
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<512xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<512xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%6] [1] : memref<512xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<512xf32> to tensor<512xf32>
    %9 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %9[] : tensor<f32>
    %reduced = linalg.reduce ins(%8 : tensor<512xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.maxnumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %10 = tensor.empty() : tensor<512xf32>
    %11 = linalg.fill ins(%extracted : f32) outs(%10 : tensor<512xf32>) -> tensor<512xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %11 : tensor<512xf32>, tensor<512xf32>) outs(%8 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %17 = arith.subf %in, %in_7 : f32
      linalg.yield %17 : f32
    } -> tensor<512xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<512xf32>) outs(%12 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = math.exp %in : f32
      linalg.yield %17 : f32
    } -> tensor<512xf32>
    %14 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst_0 into %14[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%13 : tensor<512xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %15 = linalg.fill ins(%extracted_4 : f32) outs(%10 : tensor<512xf32>) -> tensor<512xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%13, %15 : tensor<512xf32>, tensor<512xf32>) outs(%13 : tensor<512xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %17 = arith.divf %in, %in_7 : f32
      linalg.yield %17 : f32
    } -> tensor<512xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %16[0] [%6] [1] : tensor<512xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<512xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


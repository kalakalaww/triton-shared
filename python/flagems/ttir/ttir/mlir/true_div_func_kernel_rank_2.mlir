#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @true_div_func_kernel_rank_2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c512_i64 = arith.constant 512 : i64
    %c512_i32 = arith.constant 512 : i32
    %c511_i32 = arith.constant 511 : i32
    %0 = arith.extsi %arg12 : i32 to i64
    %1 = arith.addi %arg7, %c511_i32 : i32
    %2 = arith.divsi %1, %c512_i32 : i32
    %3 = arith.extsi %2 : i32 to i64
    %4 = arith.remsi %0, %3 : i64
    %5 = arith.divsi %0, %3 : i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = arith.muli %4, %c512_i64 : i64
    %8 = arith.trunci %7 : i64 to i32
    %9 = arith.index_cast %arg3 : i32 to index
    %10 = arith.index_cast %6 : i32 to index
    %11 = arith.muli %10, %9 : index
    %12 = arith.index_cast %8 : i32 to index
    %13 = arith.addi %11, %12 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [1, 512], strides: [%9, 1] : memref<*xf32> to memref<1x512xf32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x512xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1x512xf32, strided<[?, 1], offset: ?>> to memref<1x512xf32>
    %14 = bufferization.to_tensor %alloc restrict writable : memref<1x512xf32> to tensor<1x512xf32>
    %15 = arith.index_cast %arg4 : i32 to index
    %16 = arith.muli %12, %15 : index
    %17 = arith.addi %10, %16 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%17], sizes: [1, 512], strides: [1, %15] : memref<*xf32> to memref<1x512xf32, strided<[1, ?], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1x512xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x512xf32, strided<[1, ?], offset: ?>> to memref<1x512xf32>
    %18 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x512xf32> to tensor<1x512xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %18 : tensor<1x512xf32>, tensor<1x512xf32>) outs(%14 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %23 = arith.divf %in, %in_3 : f32
      linalg.yield %23 : f32
    } -> tensor<1x512xf32>
    %20 = arith.index_cast %arg5 : i32 to index
    %21 = arith.muli %10, %20 : index
    %22 = arith.addi %21, %12 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 512], strides: [%20, 1] : memref<*xf32> to memref<1x512xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %19 in writable %reinterpret_cast_2 : (tensor<1x512xf32>, memref<1x512xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}


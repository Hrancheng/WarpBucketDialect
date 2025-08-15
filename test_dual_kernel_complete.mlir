func.func @test_dual_kernel_complete(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
  %threshold = arith.constant 32 : index
  %is_short = arith.cmpi slt, %len, %threshold : index
  
  // Test 1: Short row - should use if-conversion (branch-free)
  %val1 = scf.if %is_short -> f32 {
    %short_val = memref.load %data[%len] : memref<?xf32>
    %scaled = arith.mulf %short_val, %short_val : f32
    scf.yield %scaled : f32
  } else {
    %default_val = arith.constant 1.0 : f32
    scf.yield %default_val : f32
  } {wb.divergent = true}
  
  // Test 2: Long row - should use warp-reduce
  %val2 = scf.if %is_short -> f32 {
    %default_val = arith.constant 0.0 : f32
    scf.yield %default_val : f32
  } else {
    %long_val = memref.load %data[%len] : memref<?xf32>
    %reduced = standalone.warp_reduce %long_val {kind = "add", width = 32} : f32
    scf.yield %reduced : f32
  } {wb.long_row = true}
  
  // Store results
  memref.store %val1, %result[%len] : memref<?xf32>
  memref.store %val2, %result[%len] : memref<?xf32>
  
  return
} 
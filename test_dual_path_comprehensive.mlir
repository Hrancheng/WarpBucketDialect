// Test comprehensive dual-path kernel strategy
// This demonstrates all three patterns working together

func.func @test_dual_path_comprehensive(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
  %threshold = arith.constant 32 : index
  %is_short = arith.cmpi slt, %len, %threshold : index
  
  // Test 1: Uniform branch pattern (already working)
  %val1 = scf.if %is_short -> f32 {
    %short_val = memref.load %data[%len] : memref<?xf32>
    scf.yield %short_val : f32
  } else {
    %long_val = arith.constant 0.0 : f32
    scf.yield %long_val : f32
  } {wb.uniform = true}
  
  // Test 2: Short row pattern (len ≤ T)
  %val2 = scf.if %is_short -> f32 {
    %short_val = memref.load %data[%len] : memref<?xf32>
    %scaled = arith.mulf %short_val, %short_val : f32
    scf.yield %scaled : f32
  } else {
    %default_val = arith.constant 1.0 : f32
    scf.yield %default_val : f32
  } {wb.short_row = true}
  
  // Test 3: Long row pattern (len > T)
  %val3 = scf.if %is_short -> f32 {
    %default_val = arith.constant 0.0 : f32
    scf.yield %default_val : f32
  } else {
    %long_val = memref.load %data[%len] : memref<?xf32>
    %processed = arith.mulf %long_val, %long_val : f32
    scf.yield %processed : f32
  } {wb.long_row = true}
  
  // Store results
  memref.store %val1, %result[%len] : memref<?xf32>
  memref.store %val2, %result[%len] : memref<?xf32>
  memref.store %val3, %result[%len] : memref<?xf32>
  
  return
} 
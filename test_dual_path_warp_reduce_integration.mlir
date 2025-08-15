module {
  // Test the complete dual-path kernel strategy:
  // - Short rows (len ≤ 32): Use if-conversion (select + masked)
  // - Long rows (len > 32): Use warp-stride loops + warp-reduce
  
  func.func @test_dual_path_complete(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
    %threshold = arith.constant 32 : index
    %is_short = arith.cmpi slt, %len, %threshold : index
    
    // Path 1: Short row - should use if-conversion (branch-free)
    %val1 = scf.if %is_short -> f32 {
      %short_val = memref.load %data[%len] : memref<?xf32>
      %scaled = arith.mulf %short_val, %short_val : f32
      scf.yield %scaled : f32
    } else {
      %default_val = arith.constant 1.0 : f32
      scf.yield %default_val : f32
    } {wb.divergent = true}
    
    // Path 2: Long row - should use warp-reduce
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
  
  // Test 2: More complex long row with multiple warp-reduce operations
  func.func @test_long_row_warp_reduce_complex(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
    %threshold = arith.constant 32 : index
    %is_long = arith.cmpi sgt, %len, %threshold : index
    
    %final_result = scf.if %is_long -> f32 {
      // Load multiple values for complex reduction
      %val1 = memref.load %data[%len] : memref<?xf32>
      %val2 = memref.load %data[%len] : memref<?xf32>
      
      // Sum reduction
      %sum = standalone.warp_reduce %val1 {kind = "add", width = 32} : f32
      
      // Product reduction  
      %product = standalone.warp_reduce %val2 {kind = "mul", width = 32} : f32
      
      // Combine results
      %combined = arith.addf %sum, %product : f32
      scf.yield %combined : f32
    } else {
      %default = arith.constant 0.0 : f32
      scf.yield %default : f32
    } {wb.long_row = true}
    
    memref.store %final_result, %result[%len] : memref<?xf32>
    return
  }
  
  // Test 3: Mixed data types in warp-reduce
  func.func @test_mixed_types_warp_reduce(%len: index, %int_data: memref<?xi32>, %float_data: memref<?xf32>, %result: memref<?xf32>) {
    %threshold = arith.constant 32 : index
    %is_long = arith.cmpi sgt, %len, %threshold : index  // Check if length > threshold
    
    %mixed_result = scf.if %is_long -> f32 {
      // Integer reduction
      %int_val = memref.load %int_data[%len] : memref<?xi32>
      %int_sum = standalone.warp_reduce %int_val {kind = "add", width = 32} : i32
      
      // Float reduction
      %float_val = memref.load %float_data[%len] : memref<?xf32>
      %float_sum = standalone.warp_reduce %float_val {kind = "add", width = 32} : f32
      
      // Combine results (simplified)
      %final = arith.addf %float_sum, %float_sum : f32
      scf.yield %final : f32
    } else {
      %default = arith.constant 0.0 : f32
      scf.yield %default : f32
    } {wb.long_row = true}
    
    memref.store %mixed_result, %result[%len] : memref<?xf32>
    return
  }
} 
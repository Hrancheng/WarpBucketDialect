module {
  // Threshold for determining short vs long rows
  func.func @sparse_matrix_kernel(%row_lengths: memref<?xi32>, %values: memref<?xf32>, %result: memref<?xf32>) {
    // Get thread and block information
    %thread_id = gpu.thread_id x
    %block_id = gpu.block_id x
    %block_dim = gpu.block_dim x
    
    // Calculate row index
    %row_idx = arith.addi %block_id, %thread_id : i32
    
    // Load row length
    %row_len = memref.load %row_lengths[%row_idx] : memref<?xi32>
    
    // Threshold for short vs long rows (T = 32)
    %threshold = arith.constant 32 : i32
    
    // Uniform branch: choose between short and long row strategies
    %is_short = arith.cmpi slt, %row_len, %threshold : i32
    
    // Short row path: if-conversion with masked operations
    %short_result = scf.if %is_short -> f32 {
      // Mark as short row for the pass to recognize
      scf.yield %row_len : f32
    } else {
      scf.yield %row_len : f32
    } attributes {wb.short_row = true}
    
    // Long row path: warp-stride loop with reduction
    %long_result = scf.for %i = %row_idx to %row_len step %block_dim : i32 {
      // Mark as long row for the pass to recognize
      scf.yield %i : i32
    } attributes {wb.long_row = true}
    
    // Process short rows with if-conversion
    %final_short = scf.if %is_short -> f32 {
      // Use masked operations for short rows
      %temp = standalone.masked_load %values[%row_idx], %is_short, %short_result : f32
      %sum = standalone.warp_reduce %temp : f32
      scf.yield %sum : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    
    // Process long rows with warp-stride loop
    %final_long = scf.if %is_short -> f32 {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    } else {
      // Long row processing with warp-stride loop
      %sum = arith.constant 0.0 : f32
      %warp_size = arith.constant 32 : i32
      
      // This will be converted to warp-stride loop by the pass
      %long_sum = scf.for %j = %row_idx to %row_len step %warp_size : i32 {
        %val = memref.load %values[%j] : memref<?xf32>
        %new_sum = arith.addf %sum, %val : f32
        scf.yield %new_sum : f32
      } attributes {wb.long_row = true}
      
      // Final warp reduction
      %final = standalone.warp_reduce %long_sum : f32
      scf.yield %final : f32
    }
    
    // Combine results
    %final_result = arith.addf %final_short, %final_long : f32
    
    // Store result
    memref.store %final_result, %result[%row_idx] : memref<?xf32>
    
    return
  }
} 
// Test automatic warp_reduce insertion for various reduction patterns
// This demonstrates the AutoWarpReduceInsertionPass in action

func.func @test_accumulation_loop(%len: index, %data: memref<?xf32>, %result: memref<f32>) {
  // Pattern 1: Simple accumulation loop (should trigger warp_reduce insertion)
  %sum = arith.constant 0.0 : f32
  scf.for %i = 0 to %len step 1 {
    %val = memref.load %data[%i] : memref<?xf32>
    %sum = arith.addf %sum, %val : f32
  }
  
  // Pattern 2: Product accumulation loop
  %product = arith.constant 1.0 : f32
  scf.for %j = 0 to %len step 1 {
    %val2 = memref.load %data[%j] : memref<?xf32>
    %product = arith.mulf %product, %val2 : f32
  }
  
  // Pattern 3: Integer accumulation
  %int_sum = arith.constant 0 : i32
  scf.for %k = 0 to %len step 1 {
    %int_val = arith.constant 1 : i32
    %int_sum = arith.addi %int_sum, %int_val : i32
  }
  
  // Store results
  memref.store %sum, %result[0] : memref<f32>
  memref.store %product, %result[1] : memref<f32>
  
  return
}

func.func @test_nested_reductions(%len: index, %data: memref<?xf32>, %result: memref<f32>) {
  // Pattern 4: Nested loop with accumulation
  %total = arith.constant 0.0 : f32
  
  scf.for %i = 0 to %len step 1 {
    %row_sum = arith.constant 0.0 : f32
    
    scf.for %j = 0 to %len step 1 {
      %val = memref.load %data[%i] : memref<?xf32>
      %row_sum = arith.addf %row_sum, %val : f32
    }
    
    %total = arith.addf %total, %row_sum : f32
  }
  
  memref.store %total, %result[0] : memref<f32>
  return
}

func.func @test_conditional_reduction(%len: index, %data: memref<?xf32>, %result: memref<f32>) {
  // Pattern 5: Conditional accumulation (should still trigger warp_reduce)
  %sum = arith.constant 0.0 : f32
  
  scf.for %i = 0 to %len step 1 {
    %val = memref.load %data[%i] : memref<?xf32>
    %is_positive = arith.cmpf ogt, %val, %sum : f32
    
    %sum = scf.if %is_positive -> f32 {
      %new_sum = arith.addf %sum, %val : f32
      scf.yield %new_sum : f32
    } else {
      scf.yield %sum : f32
    }
  }
  
  memref.store %sum, %result[0] : memref<f32>
  return
}

func.func @test_multiple_accumulators(%len: index, %data: memref<?xf32>, %result: memref<f32>) {
  // Pattern 6: Multiple accumulators in the same loop
  %sum = arith.constant 0.0 : f32
  %min_val = arith.constant 1000.0 : f32
  %max_val = arith.constant -1000.0 : f32
  
  scf.for %i = 0 to %len step 1 {
    %val = memref.load %data[%i] : memref<?xf32>
    
    // Sum accumulation
    %sum = arith.addf %sum, %val : f32
    
    // Min accumulation
    %min_val = arith.minf %min_val, %val : f32
    
    // Max accumulation
    %max_val = arith.maxf %max_val, %val : f32
  }
  
  // Store all results
  memref.store %sum, %result[0] : memref<f32>
  memref.store %min_val, %result[1] : memref<f32>
  memref.store %max_val, %result[2] : memref<f32>
  
  return
} 
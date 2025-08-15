module {
  // ============================================================================
  // Test 1: Simple Vector Sum - Basic accumulation pattern
  // ============================================================================
  func.func @vector_sum_kernel(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // This should trigger automatic warp_reduce insertion
    %sum = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg1[%i] : memref<?xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %sum, %arg2[] : memref<f32>
    return
  }

  // ============================================================================
  // Test 2: Matrix Row Sums - Multiple independent reductions
  // ============================================================================
  func.func @matrix_row_sums(%arg0: index, %arg1: index, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // Outer loop over rows
    scf.for %row = %c0 to %arg0 step %c1 {
      // Inner loop with accumulation - should trigger warp_reduce
      %row_sum = scf.for %col = %c0 to %arg1 step %c1 iter_args(%acc = %cst) -> (f32) {
        %val = memref.load %arg2[%row, %col] : memref<?x?xf32>
        %new_acc = arith.addf %acc, %val : f32
        scf.yield %new_acc : f32
      }
      memref.store %row_sum, %arg3[%row] : memref<?xf32>
    }
    return
  }

  // ============================================================================
  // Test 3: Dot Product - Two accumulations in sequence
  // ============================================================================
  func.func @dot_product_kernel(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // First accumulation: element-wise multiplication
    %dot = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst) -> (f32) {
      %a = memref.load %arg1[%i] : memref<?xf32>
      %b = memref.load %arg2[%i] : memref<?xf32>
      %prod = arith.mulf %a, %b : f32
      %new_acc = arith.addf %acc, %prod : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %dot, %arg3[] : memref<f32>
    return
  }

  // ============================================================================
  // Test 4: Histogram - Integer accumulation with conditional
  // ============================================================================
  func.func @histogram_kernel(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0 : i32
    
    // Initialize histogram bins
    scf.for %bin = %c0 to %arg0 step %c1 {
      memref.store %cst, %arg2[%bin] : memref<?xi32>
    }
    
    // Count occurrences - should trigger warp_reduce
    scf.for %i = %c0 to %arg0 step %c1 iter_args(%dummy = %cst) -> (i32) {
      %val = memref.load %arg1[%i] : memref<?xi32>
      %bin_idx = arith.index_cast %val : i32 to index
      %current_count = memref.load %arg2[%bin_idx] : memref<?xi32>
      %new_count = arith.addi %current_count, %c1_i32 : i32
      memref.store %new_count, %arg2[%bin_idx] : memref<?xi32>
      scf.yield %dummy : i32
    }
    return
  }

  // ============================================================================
  // Test 5: Max Pooling - Reduction with comparison
  // ============================================================================
  func.func @max_pooling_kernel(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant -3.402823e+38 : f32  // -FLT_MAX
    
    // Find maximum value - should trigger warp_reduce
    %max_val = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg1[%i] : memref<?xf32>
      %new_max = arith.maximumf %acc, %val : f32
      scf.yield %new_max : f32
    }
    
    memref.store %max_val, %arg2[] : memref<f32>
    return
  }

  // ============================================================================
  // Test 6: Complex Nested Reductions - Multiple levels
  // ============================================================================
  func.func @nested_reductions_kernel(%arg0: index, %arg1: index, %arg2: memref<?x?xf32>, %arg3: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // Outer loop with accumulation
    %outer_sum = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst) -> (f32) {
      // Inner loop with accumulation - should trigger warp_reduce
      %inner_sum = scf.for %j = %c0 to %arg1 step %c1 iter_args(%inner_acc = %cst) -> (f32) {
        %val = memref.load %arg2[%i, %j] : memref<?x?xf32>
        %new_inner_acc = arith.addf %inner_acc, %val : f32
        scf.yield %new_inner_acc : f32
      }
      %new_acc = arith.addf %acc, %inner_sum : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %outer_sum, %arg3[] : memref<f32>
    return
  }

  // ============================================================================
  // Test 7: Mixed Data Types - Different reduction types
  // ============================================================================
  func.func @mixed_reductions_kernel(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xi32>, %arg3: memref<f32>, %arg4: memref<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_f = arith.constant 0.000000e+00 : f32
    %cst_i = arith.constant 0 : i32
    
    // Float sum reduction
    %float_sum = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst_f) -> (f32) {
      %val = memref.load %arg1[%i] : memref<?xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    // Integer product reduction
    %int_prod = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst_i) -> (i32) {
      %val = memref.load %arg2[%i] : memref<?xi32>
      %new_acc = arith.muli %acc, %val : i32
      scf.yield %new_acc : i32
    }
    
    memref.store %float_sum, %arg3[] : memref<f32>
    memref.store %int_prod, %arg4[] : memref<i32>
    return
  }

  // ============================================================================
  // Test 8: Edge Cases - Single iteration, empty loops
  // ============================================================================
  func.func @edge_cases_kernel(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // Single iteration loop - should still trigger warp_reduce
    %single_sum = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg1[%i] : memref<?xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    // Loop with potential zero iterations
    %conditional_sum = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg1[%i] : memref<?xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %single_sum, %arg2[] : memref<f32>
    return
  }
} 
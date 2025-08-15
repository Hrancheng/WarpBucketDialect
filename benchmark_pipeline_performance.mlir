module {
  // ============================================================================
  // Performance Benchmarking Suite
  // ============================================================================
  
  // Benchmark 1: Large-scale vector reduction (1024 elements)
  func.func @benchmark_large_vector_reduction(%arg0: memref<1024xf32>, %arg1: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // Large accumulation loop - should show significant performance improvement
    %sum = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg0[%i] : memref<1024xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %sum, %arg1[] : memref<f32>
    return
  }

  // Benchmark 2: Matrix multiplication with reduction (32x32)
  func.func @benchmark_matrix_multiply_reduction(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // Outer loop over rows
    scf.for %row = %c0 to %c32 step %c1 {
      // Inner loop with accumulation - should trigger warp_reduce
      %row_sum = scf.for %col = %c0 to %c32 step %c1 iter_args(%acc = %cst) -> (f32) {
        %a_val = memref.load %arg0[%row, %col] : memref<32x32xf32>
        %b_val = memref.load %arg1[%row, %col] : memref<32x32xf32>
        %prod = arith.mulf %a_val, %b_val : f32
        %new_acc = arith.addf %acc, %prod : f32
        scf.yield %new_acc : f32
      }
      // Store row result
      memref.store %row_sum, %arg2[%row] : memref<32xf32>
    }
    return
  }

  // Benchmark 3: Multiple independent reductions (parallel processing)
  func.func @benchmark_parallel_reductions(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<f32>, %arg3: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // First reduction: sum of first half
    %sum_first = scf.for %i = %c0 to %c256 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg0[%i] : memref<512xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    // Second reduction: sum of second half
    %sum_second = scf.for %i = %c256 to %c512 step %c1 iter_args(%acc = %cst) -> (f32) {
      %val = memref.load %arg1[%i] : memref<512xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %sum_first, %arg2[] : memref<f32>
    memref.store %sum_second, %arg3[] : memref<f32>
    return
  }

  // Benchmark 4: Complex nested reductions (real-world scenario)
  func.func @benchmark_complex_nested_reductions(%arg0: memref<64x64xf32>, %arg1: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // Outer loop with accumulation
    %outer_sum = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %cst) -> (f32) {
      // Inner loop with accumulation - should trigger warp_reduce
      %inner_sum = scf.for %j = %c0 to %c64 step %c1 iter_args(%inner_acc = %cst) -> (f32) {
        %val = memref.load %arg0[%i, %j] : memref<64x64xf32>
        %new_inner_acc = arith.addf %inner_acc, %val : f32
        scf.yield %new_inner_acc : f32
      }
      %new_acc = arith.addf %acc, %inner_sum : f32
      scf.yield %new_acc : f32
    }
    
    memref.store %outer_sum, %arg1[] : memref<f32>
    return
  }

  // Benchmark 5: Mixed data type reductions (comprehensive test)
  func.func @benchmark_mixed_reductions(%arg0: memref<256xf32>, %arg1: memref<256xi32>, %arg2: memref<f32>, %arg3: memref<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %cst_f = arith.constant 0.000000e+00 : f32
    %cst_i = arith.constant 0 : i32
    
    // Float sum reduction
    %float_sum = scf.for %i = %c0 to %c256 step %c1 iter_args(%acc = %cst_f) -> (f32) {
      %val = memref.load %arg0[%i] : memref<256xf32>
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    
    // Integer product reduction
    %int_prod = scf.for %i = %c0 to %c256 step %c1 iter_args(%acc = %cst_i) -> (i32) {
      %val = memref.load %arg1[%i] : memref<256xi32>
      %new_acc = arith.muli %acc, %val : i32
      scf.yield %new_acc : i32
    }
    
    memref.store %float_sum, %arg2[] : memref<f32>
    memref.store %int_prod, %arg3[] : memref<i32>
    return
  }
} 
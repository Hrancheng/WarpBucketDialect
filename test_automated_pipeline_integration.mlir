module {
  // Test function with accumulation pattern that should trigger automatic warp_reduce insertion
  func.func @test_accumulation_kernel(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    // This loop should trigger automatic warp_reduce insertion
    %0 = scf.for %arg3 = %c0 to %arg0 step %c1 iter_args(%arg4 = %cst) -> (f32) {
      %1 = memref.load %arg1[%arg3] : memref<?xf32>
      %2 = arith.addf %arg4, %1 : f32
      scf.yield %2 : f32
    }
    
    memref.store %0, %arg2[] : memref<f32>
    return
  }

  // Test function with multiple accumulation patterns
  func.func @test_multiple_reductions(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<f32>, %arg3: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_one = arith.constant 1.000000e+00 : f32
    
    // Sum reduction
    %sum = scf.for %arg4 = %c0 to %arg0 step %c1 iter_args(%arg5 = %cst) -> (f32) {
      %1 = memref.load %arg1[%arg4] : memref<?xf32>
      %2 = arith.addf %arg5, %1 : f32
      scf.yield %2 : f32
    }
    
    // Product reduction
    %product = scf.for %arg6 = %c0 to %arg0 step %c1 iter_args(%arg7 = %cst_one) -> (f32) {
      %3 = memref.load %arg1[%arg6] : memref<?xf32>
      %4 = arith.mulf %arg7, %3 : f32
      scf.yield %4 : f32
    }
    
    memref.store %sum, %arg2[] : memref<f32>
    memref.store %product, %arg3[] : memref<f32>
    return
  }

  // Test function with nested loops (should only process the inner accumulation loop)
  func.func @test_nested_loops(%arg0: index, %arg1: memref<?x?xf32>, %arg2: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    
    %0 = scf.for %arg3 = %c0 to %arg0 step %c1 iter_args(%arg4 = %cst) -> (f32) {
      // Inner loop with accumulation
      %inner_result = scf.for %arg5 = %c0 to %arg0 step %c1 iter_args(%arg6 = %arg4) -> (f32) {
        %1 = memref.load %arg1[%arg3, %arg5] : memref<?x?xf32>
        %2 = arith.addf %arg6, %1 : f32
        scf.yield %2 : f32
      }
      scf.yield %inner_result : f32
    }
    
    memref.store %0, %arg2[] : memref<f32>
    return
  }
} 
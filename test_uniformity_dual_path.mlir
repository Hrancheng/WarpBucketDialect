module {
  // Test function that should trigger uniformity analysis
  func.func @test_uniformity_dual_path(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
    // This should be marked as Varying by uniformity analysis
    %thread_id = gpu.thread_id x
    
    // This should be marked as Uniform (constant)
    %threshold = arith.constant 32 : index
    
    // This should be marked as Varying (depends on thread_id)
    %is_short = arith.cmpi slt, %len, %threshold : index
    
    // This should be marked as divergent by uniformity analysis
    %val1 = scf.if %is_short -> f32 {
      %short_val = memref.load %data[%len] : memref<?xf32>
      %scaled = arith.mulf %short_val, %short_val : f32
      scf.yield %scaled : f32
    } else {
      %default_val = arith.constant 1.0 : f32
      scf.yield %default_val : f32
    }
    
    // This should also be marked as divergent
    %val2 = scf.if %is_short -> f32 {
      %default_val = arith.constant 0.0 : f32
      scf.yield %default_val : f32
    } else {
      %long_val = memref.load %data[%len] : memref<?xf32>
      %reduced = standalone.warp_reduce %long_val {kind = "add", width = 32} : f32
      scf.yield %reduced : f32
    }
    
    memref.store %val1, %result[%len] : memref<?xf32>
    memref.store %val2, %result[%len] : memref<?xf32>
    
    return
  }
} 
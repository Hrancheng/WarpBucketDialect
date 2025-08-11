module {
  func.func @test_uniformity_varying() {
    // Constants are uniform
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // GPU thread ID is varying (thread-specific)
    %thread_id = gpu.thread_id x : index
    
    // GPU lane ID is varying (thread-specific)
    %lane_id = gpu.lane_id : index
    
    // Operations with uniform inputs remain uniform
    %sum = arith.addi %c1, %c2 : i32
    
    // Operations with varying inputs become varying
    %thread_plus_const = arith.addi %thread_id, %c1 : index
    %thread_times_const = arith.muli %thread_id, %c2 : index
    
    // Operations with mixed inputs (uniform + varying) become varying
    %mixed_op = arith.addi %sum, %thread_id : index
    
    // Operations with only varying inputs remain varying
    %thread_op = arith.addi %thread_id, %lane_id : index
    
    return
  }
} 
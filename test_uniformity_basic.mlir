module {
  func.func @test_uniformity_basic() {
    // Constants are uniform
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // GPU dimensions are uniform within a warp
    %block_dim_x = gpu.block_dim x : index
    %grid_dim_x = gpu.grid_dim x : index
    
    // Thread ID is varying
    %thread_id = gpu.thread_id x : index
    
    // Operations with uniform inputs remain uniform
    %sum = arith.addi %c1, %c2 : i32
    %prod = arith.muli %block_dim_x, %grid_dim_x : index
    
    // Operations with varying inputs become varying
    %thread_plus_const = arith.addi %thread_id, %c1 : index
    %thread_times_const = arith.muli %thread_id, %c2 : index
    
    return
  }
} 
module {
  func.func @test_gpu_uniformity() -> index {
    // Real GPU operations
    %thread_id_x = gpu.thread_id x
    %thread_id_y = gpu.thread_id y
    %block_dim_x = gpu.block_dim x
    %block_dim_y = gpu.block_dim y
    
    // Operations with varying inputs should become varying
    %varying_sum = arith.addi %thread_id_x, %thread_id_y : index
    
    // Operations with uniform inputs should remain uniform
    %uniform_sum = arith.addi %block_dim_x, %block_dim_y : index
    
    // Mixed operations
    %mixed = arith.addi %varying_sum, %uniform_sum : index
    
    return %mixed : index
  }
} 
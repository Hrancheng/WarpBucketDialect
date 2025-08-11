module {
  func.func @test_uniformity_branches() {
    // Constants are uniform
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    
    // Thread ID is varying
    %thread_id = gpu.thread_id x : index
    
    // This condition is varying (thread-specific)
    %varying_cond = arith.cmpi slt, %thread_id, %c10 : i1
    
    // This condition is uniform (same for all threads in warp)
    %uniform_cond = arith.cmpi slt, %c1, %c10 : i1
    
    // Branch with varying condition - will be marked as divergent
    cf.cond_br %varying_cond, ^bb1, ^bb2
    
  ^bb1:
    // This block will be executed by some threads
    return
    
  ^bb2:
    // This block will be executed by other threads
    return
  }
} 
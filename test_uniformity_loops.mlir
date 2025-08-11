module {
  func.func @test_uniformity_loops() {
    // Constants are uniform
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    
    // Thread ID is varying
    %thread_id = gpu.thread_id x : index
    
    // This loop has uniform bounds - will not be marked as varying_exit
    scf.for %i = %c0 to %c32 step %c1 {
      // Loop body
    }
    
    // This loop has varying upper bound - will be marked as varying_exit
    scf.for %j = %c0 to %thread_id step %c1 {
      // Loop body with varying exit condition
    }
    
    // This loop has varying step - will be marked as varying_exit
    scf.for %k = %c0 to %c32 step %thread_id {
      // Loop body with varying step
    }
    
    return
  }
} 
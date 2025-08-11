module {
  // Simulate varying values by using function arguments that we'll mark as varying
  func.func @test_varying_simulation(%thread_id : i32, %lane_id : i32, %uniform_val : i32) -> i32 {
    // These should be marked as Varying by our analysis
    %varying_1 = arith.addi %thread_id, %uniform_val : i32
    %varying_2 = arith.addi %lane_id, %uniform_val : i32
    
    // This should become Varying due to varying inputs
    %result = arith.addi %varying_1, %varying_2 : i32
    
    return %result : i32
  }
} 
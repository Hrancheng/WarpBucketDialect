module {
  func.func @test_uniformity_varying_simulated(%arg: i32) -> i32 {
    // Constants are uniform
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // Function argument is uniform (kernel parameter)
    // In a real GPU scenario, this would be like a thread-specific value
    
    // Operations with uniform inputs remain uniform
    %sum = arith.addi %c1, %c2 : i32
    %prod = arith.muli %c1, %c2 : i32
    
    // Operations with uniform inputs remain uniform
    %result = arith.addi %arg, %sum : i32
    
    return %result : i32
  }
} 
module {
  func.func @test_uniformity_simple_branch(%arg: i32) -> i32 {
    // Constants are uniform
    %c10 = arith.constant 10 : i32
    
    // Simple arithmetic operations
    %sum = arith.addi %arg, %c10 : i32
    
    return %sum : i32
  }
} 
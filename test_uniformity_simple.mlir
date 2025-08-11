module {
  func.func @test_uniformity_simple(%arg: i32) -> i32 {
    // Constants are uniform
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // Operations with uniform inputs remain uniform
    %sum = arith.addi %c1, %c2 : i32
    %prod = arith.muli %c1, %c2 : i32
    
    // Function arguments are uniform (kernel parameters)
    // %arg is already a function parameter
    
    // Operations with uniform inputs remain uniform
    %result = arith.addi %arg, %sum : i32
    
    return %result : i32
  }
} 
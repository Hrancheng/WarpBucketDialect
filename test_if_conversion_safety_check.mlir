module {
  func.func @test_safety_rejection(%cond: i1, %x: i32) -> i32 {
    // This should be rejected because it has a function call
    %result = scf.if %cond -> i32 {
      %temp = arith.addi %x, %x : i32
      // Note: In real MLIR, this would be a function call
      // For testing, we'll use a placeholder operation
      scf.yield %temp : i32
    } else {
      %temp2 = arith.constant 42 : i32
      scf.yield %temp2 : i32
    }
    
    return %result : i32
  }
} 
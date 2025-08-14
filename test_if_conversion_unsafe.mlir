module {
  func.func @helper_function() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
  
  func.func @test_unsafe_operations(%cond: i1, %x: i32) -> i32 {
    // This should be rejected because it has a function call
    %result = scf.if %cond -> i32 {
      %temp = arith.addi %x, %x : i32
      scf.yield %temp : i32
    } else {
      %call = func.call @helper_function() : () -> i32
      scf.yield %call : i32
    }
    
    return %result : i32
  }
} 
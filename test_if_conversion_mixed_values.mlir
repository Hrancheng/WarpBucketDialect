module {
  func.func @test_mixed_values(%cond: i1, %x: i32) -> (i32, i1) {
    // Pre-compute some values outside the if
    %pre_computed = arith.addi %x, %x : i32
    %constant_val = arith.constant 42 : i32
    
    // This divergent branch mixes external and internal values
    %result1, %result2 = scf.if %cond -> (i32, i1) {
      %internal = arith.muli %x, %x : i32
      scf.yield %internal, %cond : i32, i1
    } else {
      %temp = arith.addi %pre_computed, %constant_val : i32
      scf.yield %temp, %cond : i32, i1
    }
    
    return %result1, %result2 : i32, i1
  }
} 
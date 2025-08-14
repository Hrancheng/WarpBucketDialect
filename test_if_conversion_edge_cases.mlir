module {
  func.func @test_edge_cases(%cond: i1, %x: i32) -> (i32, i1) {
    // Test 1: Empty then branch (just yield)
    %result1, %result2 = scf.if %cond -> (i32, i1) {
      scf.yield %x, %cond : i32, i1
    } else {
      %temp = arith.addi %x, %x : i32
      %neg = arith.cmpi eq, %temp, %x : i32
      scf.yield %temp, %neg : i32, i1
    }
    
    return %result1, %result2 : i32, i1
  }
} 
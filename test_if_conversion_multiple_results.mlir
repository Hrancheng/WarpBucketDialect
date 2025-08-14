module {
  func.func @test_multiple_results(%cond: i1) -> (i1, i32) {
    // This divergent branch should be converted to predicated operations
    %result1, %result2 = scf.if %cond -> (i1, i32) {
      %a = arith.addi %cond, %cond : i1
      %b = arith.constant 100 : i32
      scf.yield %a, %b : i1, i32
    } else {
      %c = arith.constant 1 : i1
      %d = arith.constant 42 : i32
      scf.yield %c, %d : i1, i32
    }
    
    return %result1, %result2 : i1, i32
  }
} 
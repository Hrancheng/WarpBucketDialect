module {
  func.func @test_nested_types(%cond: i1, %x: i32, %y: f32) -> (i32, f32, i1) {
    // This divergent branch has nested operations with different types
    %result1, %result2, %result3 = scf.if %cond -> (i32, f32, i1) {
      %temp1 = arith.addi %x, %x : i32
      %temp2 = arith.muli %temp1, %temp1 : i32
      %temp3 = arith.cmpi eq, %temp2, %x : i32
      scf.yield %temp2, %y, %temp3 : i32, f32, i1
    } else {
      %temp4 = arith.constant 100 : i32
      %temp5 = arith.addf %y, %y : f32
      %temp6 = arith.cmpi eq, %temp4, %x : i32
      scf.yield %temp4, %temp5, %temp6 : i32, f32, i1
    }
    
    return %result1, %result2, %result3 : i32, f32, i1
  }
} 
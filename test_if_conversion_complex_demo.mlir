func.func @test_if_conversion_complex_demo(%cond: i1, %a: i32, %b: i32) -> i32 {
  // This should be converted to predicated operations
  %result = scf.if %cond -> i32 {
    %sum = arith.addi %a, %b : i32
    %product = arith.muli %sum, %a : i32
    scf.yield %product : i32
  } else {
    %diff = arith.subi %a, %b : i32
    %squared = arith.muli %diff, %diff : i32
    scf.yield %squared : i32
  } {wb.divergent = true}
  
  return %result : i32
} 
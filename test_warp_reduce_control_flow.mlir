module {
  func.func @test_warp_reduce_control_flow(%cond: i1, %x: i32) -> i32 {
    // Warp reduction in conditional branch
    %result = scf.if %cond -> i32 {
      %temp = standalone.warp_reduce %x : i32
      scf.yield %temp : i32
    } else {
      %temp2 = arith.constant 42 : i32
      %temp3 = standalone.warp_reduce %temp2 : i32
      scf.yield %temp3 : i32
    }
    
    return %result : i32
  }
} 
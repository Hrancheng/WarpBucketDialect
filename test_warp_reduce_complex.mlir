module {
  func.func @test_warp_reduce_complex(%x: i32, %y: i32) -> i32 {
    // Complex expression as input to warp reduction
    %temp1 = arith.addi %x, %y : i32
    %temp2 = arith.muli %temp1, %x : i32
    %temp3 = arith.subi %temp2, %y : i32
    
    // Warp reduce the complex expression result
    %result = standalone.warp_reduce %temp3 : i32
    
    return %result : i32
  }
} 
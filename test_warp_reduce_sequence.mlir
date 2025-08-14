module {
  func.func @test_warp_reduce_sequence(%x: i32, %y: i32) -> (i32, i32) {
    // First warp reduction
    %result1 = standalone.warp_reduce %x : i32
    
    // Second warp reduction (should work independently)
    %result2 = standalone.warp_reduce %y : i32
    
    // Use results in arithmetic operations
    %sum = arith.addi %result1, %result2 : i32
    %product = arith.muli %result1, %result2 : i32
    
    return %sum, %product : i32, i32
  }
} 
module {
  func.func @test_warp_reduce_edge_cases(%x: i32) -> (i32, i32) {
    // Edge case 1: Warp reduce the same value multiple times
    %result1 = standalone.warp_reduce %x : i32
    %result2 = standalone.warp_reduce %x : i32
    
    // Edge case 2: Warp reduce a warp reduced result
    %nested = standalone.warp_reduce %result1 : i32
    
    // Edge case 3: Use warp reduce in a simple expression
    %final = arith.addi %result2, %nested : i32
    
    return %final, %nested : i32, i32
  }
} 
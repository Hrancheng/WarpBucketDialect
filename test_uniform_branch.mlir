module {
  func.func @test_uniform_branch(%block_size: i32, %warp_size: i32) {
    // Calculate if this block should use short or long strategy
    %use_short = arith.cmpi slt, %block_size, %warp_size : i32
    
    // Uniform branch: all threads in warp follow same path
    %result = scf.if %use_short -> i32 {
      // Short strategy: if-conversion with masked operations
      %temp = arith.constant 1 : i32
      scf.yield %temp : i32
    } else {
      // Long strategy: warp-stride loop with reduction
      %temp = arith.constant 2 : i32
      scf.yield %temp : i32
    } attributes {wb.uniform = true}
    
    // Use the result
    %final = arith.addi %result, %result : i32
    
    return
  }
} 
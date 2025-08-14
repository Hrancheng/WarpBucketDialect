module {
  func.func @test_warp_reduce_inputs(%x: i32) -> (i32, i32, i32) {
    // Warp reduce a function argument
    %arg_result = standalone.warp_reduce %x : i32
    
    // Warp reduce a constant
    %const = arith.constant 100 : i32
    %const_result = standalone.warp_reduce %const : i32
    
    // Warp reduce a computed value
    %computed = arith.addi %x, %x : i32
    %computed_result = standalone.warp_reduce %computed : i32
    
    return %arg_result, %const_result, %computed_result : i32, i32, i32
  }
} 
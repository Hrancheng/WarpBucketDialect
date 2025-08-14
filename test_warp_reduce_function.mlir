module {
  func.func @helper_function(%val: i32) -> i32 {
    %result = arith.addi %val, %val : i32
    return %result : i32
  }
  
  func.func @test_warp_reduce_function(%x: i32) -> i32 {
    // Call helper function and use result in warp reduction
    %temp = func.call @helper_function(%x) : (i32) -> i32
    %result = standalone.warp_reduce %temp : i32
    
    return %result : i32
  }
} 
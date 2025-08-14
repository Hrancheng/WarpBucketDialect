module {
  func.func @test_warp_reduce_mixed(%x: i32, %y: f32) -> (i32, f32) {
    // Integer reduction with NVVM width (32)
    %int_result = standalone.warp_reduce %x : i32
    
    // Float reduction with ROCDL width (64)
    %float_result = standalone.warp_reduce %y : f32
    
    return %int_result, %float_result : i32, f32
  }
} 
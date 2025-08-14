module {
  func.func @test_warp_reduce_types(%x: i32, %y: f32, %z: i64) -> (i32, f32, i64) {
    // Test different integer types
    %int_result = standalone.warp_reduce %x : i32
    
    // Test floating point types
    %float_result = standalone.warp_reduce %y : f32
    
    // Test larger integer types
    %long_result = standalone.warp_reduce %z : i64
    
    return %int_result, %float_result, %long_result : i32, f32, i64
  }
} 
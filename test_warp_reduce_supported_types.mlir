module {
  // Test 1: Basic warp_reduce with i32 (supported by gpu.shuffle)
  func.func @test_warp_reduce_i32(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x : i32
    return %result : i32
  }
  
  // Test 2: Warp_reduce with f32 (supported by gpu.shuffle)
  func.func @test_warp_reduce_f32(%x: f32) -> f32 {
    %result = standalone.warp_reduce %x : f32
    return %result : f32
  }
  
  // Test 3: Warp_reduce with i64 (supported by gpu.shuffle)
  func.func @test_warp_reduce_i64(%x: i64) -> i64 {
    %result = standalone.warp_reduce %x : i64
    return %result : i64
  }
  
  // Test 4: Warp_reduce with f64 (supported by gpu.shuffle)
  func.func @test_warp_reduce_f64(%x: f64) -> f64 {
    %result = standalone.warp_reduce %x : f64
    return %result : f64
  }
} 
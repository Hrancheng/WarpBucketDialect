module {
  // Test 1: Basic addition reduction (default)
  func.func @test_warp_reduce_add(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x : i32
    return %result : i32
  }
  
  // Test 2: Explicit addition reduction
  func.func @test_warp_reduce_add_explicit(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x {kind = "add", width = 32} : i32
    return %result : i32
  }
  
  // Test 3: Multiplication reduction
  func.func @test_warp_reduce_mul(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x {kind = "mul", width = 32} : i32
    return %result : i32
  }
  
  // Test 4: AND reduction
  func.func @test_warp_reduce_and(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x {kind = "and", width = 32} : i32
    return %result : i32
  }
  
  // Test 5: Float addition reduction
  func.func @test_warp_reduce_float(%x: f32) -> f32 {
    %result = standalone.warp_reduce %x {kind = "add", width = 32} : f32
    return %result : f32
  }
} 
module {
  // Simple test: Just warp-reduce operations
  func.func @test_warp_reduce_simple(%x: f32) -> f32 {
    %result = standalone.warp_reduce %x {kind = "add", width = 32} : f32
    return %result : f32
  }
  
  // Test with integer
  func.func @test_warp_reduce_int(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x {kind = "add", width = 32} : i32
    return %result : i32
  }
  
  // Test with multiplication
  func.func @test_warp_reduce_mul(%x: f32) -> f32 {
    %result = standalone.warp_reduce %x {kind = "mul", width = 32} : f32
    return %result : f32
  }
} 
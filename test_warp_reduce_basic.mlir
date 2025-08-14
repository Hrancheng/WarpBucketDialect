module {
  func.func @test_warp_reduce_basic(%x: i32) -> i32 {
    // Basic warp reduction operation
    %result = standalone.warp_reduce %x : i32
    return %result : i32
  }
} 
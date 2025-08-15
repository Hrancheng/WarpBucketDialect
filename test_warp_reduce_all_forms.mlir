module {
  // Test 1: Basic warp_reduce with integer
  func.func @test_warp_reduce_int(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x : i32
    return %result : i32
  }
  
  // Test 2: Warp_reduce with float
  func.func @test_warp_reduce_float(%x: f32) -> f32 {
    %result = standalone.warp_reduce %x : f32
    return %result : f32
  }
  
  // Test 3: Warp_reduce with index type
  func.func @test_warp_reduce_index(%x: index) -> index {
    %result = standalone.warp_reduce %x : index
    return %result : index
  }
  
  // Test 4: Warp_reduce with vector type
  func.func @test_warp_reduce_vector(%x: vector<4xf32>) -> vector<4xf32> {
    %result = standalone.warp_reduce %x : vector<4xf32>
    return %result : vector<4xf32>
  }
  
  // Test 5: Warp_reduce with custom attributes (these would be parsed but not used in current implementation)
  func.func @test_warp_reduce_with_attrs(%x: i32) -> i32 {
    %result = standalone.warp_reduce %x {kind = "add", width = 32} : i32
    return %result : i32
  }
} 
module {
  func.func @test_masked_operations(%mask: i1, %data: memref<?xf32>, %result: memref<?xf32>, %idx: index) {
    // Masked load: only load when mask is true, otherwise use passthrough
    %passthrough = arith.constant 0.0 : f32
    %loaded = standalone.masked_load %data[%idx], %mask, %passthrough : f32
    
    // Process the loaded value
    %processed = arith.addf %loaded, %loaded : f32
    
    // Masked store: only store when mask is true
    standalone.masked_store %processed, %result[%idx], %mask
    
    return
  }
} 
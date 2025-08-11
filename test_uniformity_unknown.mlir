module {
  // Function that takes a pointer - this creates "Unknown" values
  func.func @test_uniformity_unknown(%ptr : memref<10xi32>, %index : index) -> i32 {
    // %ptr is Unknown - we don't know if all threads point to same memory
    // %index is Unknown - we don't know if all threads use same index
    
    // Load from memory - this creates Unknown result
    %loaded_value = memref.load %ptr[%index] : memref<10xi32>
    
    // %loaded_value is Unknown - depends on memory content
    
    // Mix with uniform values
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %loaded_value, %c1 : i32
    
    // %sum is Unknown (Unknown + Uniform = Unknown)
    
    return %sum : i32
  }
} 
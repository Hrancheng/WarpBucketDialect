module {
  func.func @test_uniformity_varying_simple() {
    // GPU thread ID is varying (thread-specific)
    %thread_id = "gpu.thread_id"() {dimension = "x"} : () -> index
    
    return
  }
} 
module {
  func.func @test_if_conversion_success() -> index {
    %thread_id = gpu.thread_id x
    %varying_cond = arith.cmpi eq, %thread_id, %thread_id : index
    
    // This divergent branch should be converted to predicated operations
    %result = scf.if %varying_cond -> index {
      %a = arith.addi %thread_id, %thread_id : index
      scf.yield %a : index
    } else {
      %b = arith.constant 42 : index
      scf.yield %b : index
    }
    
    return %result : index
  }
} 
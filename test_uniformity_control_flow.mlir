module {
  func.func @test_branch_divergence(%cond : i1) -> index {
    // This condition varies across threads (simulating divergent branch)
    %thread_id = gpu.thread_id x
    %varying_cond = arith.cmpi eq, %thread_id, %thread_id : index
    
    // This should be marked as divergent because condition varies
    %result = scf.if %varying_cond -> index {
      // True branch - all threads with varying_cond = true
      %a = arith.addi %thread_id, %thread_id : index
      scf.yield %a : index
    } else {
      // False branch - all threads with varying_cond = false  
      %b = arith.constant 42 : index
      scf.yield %b : index
    }
    
    return %result : index
  }
} 
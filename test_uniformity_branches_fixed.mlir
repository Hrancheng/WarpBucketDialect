module {
  func.func @test_uniformity_branches_fixed(%arg: i32) -> i32 {
    // Constants are uniform
    %c10 = arith.constant 10 : i32
    
    // This condition is uniform (same for all threads in warp)
    %uniform_cond = arith.cmpi slt, %arg, %c10 : i1
    
    // Branch with uniform condition - will NOT be marked as divergent
    cf.cond_br %uniform_cond, ^bb1, ^bb2
    
  ^bb1:
    // This block will be executed by all threads
    %result1 = arith.addi %arg, %c10 : i32
    cf.br ^bb3(%result1 : i32)
    
  ^bb2:
    // This block will also be executed by all threads
    %result2 = arith.subi %arg, %c10 : i32
    cf.br ^bb3(%result2 : i32)
    
  ^bb3(%final_result: i32):
    return %final_result : i32
  }
} 
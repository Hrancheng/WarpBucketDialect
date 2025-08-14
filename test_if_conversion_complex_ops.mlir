module {
  func.func @test_complex_operations(%x: index, %y: index) -> index {
    %thread_id = gpu.thread_id x
    %varying_cond = arith.cmpi eq, %thread_id, %thread_id : index
    
    // This divergent branch has complex nested operations
    %result = scf.if %varying_cond -> index {
      %temp1 = arith.addi %x, %y : index
      %temp2 = arith.muli %temp1, %temp1 : index
      %temp3 = arith.subi %temp2, %x : index
      scf.yield %temp3 : index
    } else {
      %temp4 = arith.muli %x, %x : index
      %temp5 = arith.addi %temp4, %y : index
      scf.yield %temp5 : index
    }
    
    return %result : index
  }
} 
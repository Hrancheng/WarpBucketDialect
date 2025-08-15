module {
  func.func @test_if_conversion_success() -> index {
    %true = arith.constant true
    %c42 = arith.constant 42 : index
    %0 = gpu.thread_id  x
    %1 = scf.if %true -> (index) {
      %2 = arith.addi %0, %0 : index
      scf.yield %2 : index
    } else {
      scf.yield %c42 : index
    } {wb.divergent = true}
    return %1 : index
  }
} 
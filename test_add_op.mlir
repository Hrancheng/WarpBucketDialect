module {
  func.func @test_add() {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %result = standalone.add %c1, %c2 : i32
    return
  }
} 
module {
  // Simple test for auto warp_reduce insertion
  func.func @test_simple(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sum_init = arith.constant 0.0 : f32
    
    %sum_final = scf.for %i = %c0 to %len step %c1 iter_args(%sum_iter = %sum_init) -> f32 {
      %val = memref.load %data[%i] : memref<?xf32>
      %sum_next = arith.addf %sum_iter, %val : f32
      scf.yield %sum_next : f32
    }
    
    memref.store %sum_final, %result[%c0] : memref<?xf32>
    return
  }
} 
module {
  func.func @test_warp_stride_loop(%start: index, %end: index, %data: memref<?xf32>) {
    // Warp size
    %warp_size = arith.constant 32 : index
    
    // This loop will be converted to warp-stride loop by the pass
    %sum = scf.for %i = %start to %end step %warp_size : index {
      // Each thread processes elements with stride equal to warp size
      %val = memref.load %data[%i] : memref<?xf32>
      
      // Accumulate values
      %new_sum = arith.addf %sum, %val : f32
      
      scf.yield %new_sum : f32
    } attributes {wb.long_row = true}
    
    // Final warp reduction
    %final = standalone.warp_reduce %sum : f32
    
    return
  }
} 
// RUN: standalone-opt --dual-path-kernel %s -o - | FileCheck %s

// Test basic dual-path kernel functionality
func.func @test_dual_path_simple(%len: index, %data: memref<?xf32>, %result: memref<?xf32>) {
  %threshold = arith.constant 32 : index
  %is_short = arith.cmpi slt, %len, %threshold : index
  
  // Test with a simple scf.if that should be converted
  %val = scf.if %is_short -> f32 {
    %short_val = memref.load %data[%len] : memref<?xf32>
    scf.yield %short_val : f32
  } else {
    %long_val = arith.constant 0.0 : f32
    scf.yield %long_val : f32
  } {wb.uniform = true}
  
  memref.store %val, %result[%len] : memref<?xf32>
  func.return
} 
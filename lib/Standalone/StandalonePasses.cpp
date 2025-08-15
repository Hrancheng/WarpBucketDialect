//===- StandalonePasses.cpp - Standalone dialect pass registration --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandalonePasses.h"
#include "Standalone/StandaloneDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Forward declarations for passes
namespace mlir {
namespace standalone {
std::unique_ptr<Pass> createLowerStandaloneToArithPass();
std::unique_ptr<Pass> createLowerArithToGPUPass();
std::unique_ptr<Pass> createUniformityAnalysisPass();
std::unique_ptr<Pass> createIfConversionPass();
std::unique_ptr<Pass> createLowerWarpReduceToGPUPass();
std::unique_ptr<Pass> createDualPathKernelPass();
std::unique_ptr<Pass> createAutoWarpReduceInsertionPass();
} // namespace standalone
} // namespace mlir

namespace mlir {
namespace standalone {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerStandalonePasses() {
  // Register all passes
  PassPipelineRegistration<>(
      "standalone-lower-to-arith",
      "Lower Standalone dialect to Arith dialect",
      [](OpPassManager &pm) {
        pm.addPass(createLowerStandaloneToArithPass());
      });

  PassPipelineRegistration<>(
      "standalone-lower-to-gpu",
      "Lower Arith dialect to GPU dialect",
      [](OpPassManager &pm) {
        pm.addPass(createLowerArithToGPUPass());
      });

  PassPipelineRegistration<>(
      "uniformity-analysis",
      "Analyze uniformity of values across GPU warps",
      [](OpPassManager &pm) {
        pm.addPass(createUniformityAnalysisPass());
      });

  PassPipelineRegistration<>(
      "if-conversion",
      "Convert divergent branches to predicated operations",
      [](OpPassManager &pm) {
        pm.addPass(createIfConversionPass());
      });

  PassPipelineRegistration<>(
      "lower-warp-reduce-to-gpu",
      "Lower warp reduce operations to GPU shuffle operations",
      [](OpPassManager &pm) {
        pm.addPass(createLowerWarpReduceToGPUPass());
      });

  PassPipelineRegistration<>(
      "dual-path-kernel",
      "Implement dual-path kernel strategy for sparse operations",
      [](OpPassManager &pm) {
        pm.addPass(createDualPathKernelPass());
      });

  PassPipelineRegistration<>(
      "auto-warp-reduce-insertion",
      "Automatically detect and insert warp_reduce operations",
      [](OpPassManager &pm) {
        pm.addPass(createAutoWarpReduceInsertionPass());
      });

  // Complete pipeline for branch-free transformation
  PassPipelineRegistration<>(
      "standalone-branch-free-pipeline",
      "Complete pipeline: Uniformity Analysis → If-Conversion → Warp-Reduce Lowering",
      [](OpPassManager &pm) {
        // Step 1: Analyze uniformity and mark divergent branches
        pm.addPass(createUniformityAnalysisPass());
        
        // Step 2: Convert divergent branches to predicated operations
        pm.addPass(createIfConversionPass());
        
        // Step 3: Lower warp-reduce operations to GPU shuffle
        pm.addPass(createLowerWarpReduceToGPUPass());
        
        // Step 4: Apply dual-path kernel strategy
        pm.addPass(createDualPathKernelPass());
      });

  // NEW: Complete automated pipeline with automatic warp_reduce insertion
  PassPipelineRegistration<>(
      "standalone-automated-pipeline",
      "Complete automated pipeline: Auto Warp-Reduce Insertion → Warp-Reduce Lowering → Uniformity Analysis → If-Conversion → Dual-Path Kernel",
      [](OpPassManager &pm) {
        // Step 1: Automatically detect and insert warp_reduce operations
        pm.addPass(createAutoWarpReduceInsertionPass());
        
        // Step 2: Lower warp-reduce operations to GPU shuffle operations
        pm.addPass(createLowerWarpReduceToGPUPass());
        
        // Step 3: Analyze uniformity and mark divergent branches
        pm.addPass(createUniformityAnalysisPass());
        
        // Step 4: Convert divergent branches to predicated operations
        pm.addPass(createIfConversionPass());
        
        // Step 5: Apply dual-path kernel strategy
        pm.addPass(createDualPathKernelPass());
      });

  // GPU lowering pipeline
  PassPipelineRegistration<>(
      "standalone-gpu-lowering",
      "Lower Standalone dialect to GPU dialect",
      [](OpPassManager &pm) {
        pm.addPass(createLowerStandaloneToArithPass());
        pm.addPass(createLowerArithToGPUPass());
      });

  // NEW: Warp-reduce automation pipeline
  PassPipelineRegistration<>(
      "standalone-warp-reduce-automation",
      "Automated warp-reduce pipeline: Insert → Lower to GPU",
      [](OpPassManager &pm) {
        // Step 1: Automatically detect and insert warp_reduce operations
        pm.addPass(createAutoWarpReduceInsertionPass());
        
        // Step 2: Lower warp-reduce operations to GPU shuffle operations
        pm.addPass(createLowerWarpReduceToGPUPass());
      });
}

} // namespace standalone
} // namespace mlir

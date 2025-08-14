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
}

} // namespace standalone
} // namespace mlir

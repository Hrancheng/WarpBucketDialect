//===- StandalonePasses.h - Standalone dialect passes --------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_STANDALONE_STANDALONEPASSES_H
#define MLIR_STANDALONE_STANDALONEPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Create a pass for lowering to operations in the `Arith` dialect.
std::unique_ptr<Pass> createLowerStandaloneToArithPass();

/// Create a pass for lowering Arith operations to GPU operations.
std::unique_ptr<Pass> createLowerArithToGPUPass();

/// Create a pass for analyzing uniformity of values across GPU warps.
std::unique_ptr<Pass> createUniformityAnalysisPass();

/// Create a pass for converting divergent branches to predicated operations.
std::unique_ptr<Pass> createIfConversionPass();

/// Create a pass for lowering warp reduce operations to GPU shuffle operations.
std::unique_ptr<Pass> createLowerWarpReduceToGPUPass();

/// Create a pass for implementing dual-path kernel strategy.
std::unique_ptr<Pass> createDualPathKernelPass();

/// Create a pass for automatically inserting warp_reduce operations.
std::unique_ptr<Pass> createAutoWarpReduceInsertionPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register all passes in the standalone dialect.
void registerStandalonePasses();

} // namespace standalone
} // namespace mlir

#endif // MLIR_STANDALONE_STANDALONEPASSES_H

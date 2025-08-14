//===- IfConversion.cpp - Convert divergent branches to predicated ops ----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandalonePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct IfConversionPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp, PatternRewriter &rewriter) const override {
    // Check if this is a divergent branch (should have been marked by uniformity analysis)
    if (!ifOp->hasAttr("wb.divergent")) {
      return failure();
    }

    llvm::errs() << "IfConversion: Processing divergent branch\n";

    // Step 1: Get the condition and results
    Value condition = ifOp.getCondition();
    auto results = ifOp.getResults();
    
    if (results.empty()) {
      llvm::errs() << "IfConversion: No results to convert\n";
      return failure();
    }

    // Step 2: Get yield values from both branches
    SmallVector<Value> thenYields, elseYields;
    
    // Extract from then branch
    if (ifOp.getThenRegion().hasOneBlock()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      for (auto yieldOp : thenBlock.getOps<scf::YieldOp>()) {
        thenYields = yieldOp.getOperands();
        break;
      }
    }
    
    // Extract from else branch
    if (ifOp.getElseRegion().hasOneBlock()) {
      Block &elseBlock = ifOp.getElseRegion().front();
      for (auto yieldOp : elseBlock.getOps<scf::YieldOp>()) {
        elseYields = yieldOp.getOperands();
        break;
      }
    }

    if (thenYields.size() != results.size() || elseYields.size() != results.size()) {
      llvm::errs() << "IfConversion: Mismatch in yield operand counts\n";
      return failure();
    }

    // Step 3: Check for non-speculatable operations (barrier/atomic/call/exceptional side effects)
    if (hasNonSpeculatableOperations(ifOp)) {
      llvm::errs() << "IfConversion: Skipping if with non-speculatable operations\n";
      return failure();
    }

    // Step 4: Clone operations to parent block before the if
    rewriter.setInsertionPoint(ifOp);
    
    // Create IRMapping for proper value mapping
    IRMapping thenMap, elseMap;
    
    // Map external values (function arguments, block arguments, etc.) to themselves
    // This ensures that values defined outside the if regions can be found
    for (Value arg : ifOp->getOperands()) {
      thenMap.map(arg, arg);
      elseMap.map(arg, arg);
    }
    
    // Also map all values that appear in yield operations to themselves
    // This handles cases where yield operands are function arguments or other external values
    for (Value thenVal : thenYields) {
      if (!thenMap.contains(thenVal)) {
        thenMap.map(thenVal, thenVal);
      }
    }
    for (Value elseVal : elseYields) {
      if (!elseMap.contains(elseVal)) {
        elseMap.map(elseVal, elseVal);
      }
    }
    
    // Clone then branch operations with proper mapping
    if (ifOp.getThenRegion().hasOneBlock()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      for (Operation &op : thenBlock.without_terminator()) {
        // Clone the operation with mapping
        Operation *cloned = rewriter.clone(op, thenMap);
        llvm::errs() << "IfConversion: Cloned then operation: " << op.getName() << " -> " << cloned->getName() << "\n";
        
        // Map the original results to cloned results
        for (auto [origResult, clonedResult] : llvm::zip(op.getResults(), cloned->getResults())) {
          thenMap.map(origResult, clonedResult);
          llvm::errs() << "IfConversion: Mapped then result: " << origResult << " -> " << clonedResult << "\n";
        }
      }
    }

    // Clone else branch operations with proper mapping
    if (ifOp.getElseRegion().hasOneBlock()) {
      Block &elseBlock = ifOp.getElseRegion().front();
      for (Operation &op : elseBlock.without_terminator()) {
        // Clone the operation with mapping
        Operation *cloned = rewriter.clone(op, elseMap);
        llvm::errs() << "IfConversion: Cloned else operation: " << op.getName() << " -> " << cloned->getName() << "\n";
        
        // Map the original results to cloned results
        for (auto [origResult, clonedResult] : llvm::zip(op.getResults(), cloned->getResults())) {
          elseMap.map(origResult, clonedResult);
          llvm::errs() << "IfConversion: Mapped else result: " << origResult << " -> " << clonedResult << "\n";
        }
      }
    }

    // Step 5: Create select operations for each result
    SmallVector<Value> newResults;
    for (auto [thenVal, elseVal, result] : llvm::zip(thenYields, elseYields, results)) {
      // Use the IRMapping to find the cloned values
      Value thenCloned = thenMap.lookup(thenVal);
      Value elseCloned = elseMap.lookup(elseVal);
      
      // If we didn't find cloned values, the values might already be defined outside the if
      if (!thenCloned) {
        // Check if the value is defined outside the if statement
        if (thenVal.getDefiningOp() && thenVal.getDefiningOp()->getParentOp() != ifOp) {
          thenCloned = thenVal; // Use the original value
          llvm::errs() << "IfConversion: Using original then value (defined outside if): " << thenVal << "\n";
        }
      }
      
      if (!elseCloned) {
        // Check if the value is defined outside the if statement
        if (elseVal.getDefiningOp() && elseVal.getDefiningOp()->getParentOp() != ifOp) {
          elseCloned = elseVal; // Use the original value
          llvm::errs() << "IfConversion: Using original else value (defined outside if): " << elseVal << "\n";
        }
      }
      
      if (!thenCloned || !elseCloned) {
        llvm::errs() << "IfConversion: Failed to find cloned values\n";
        llvm::errs() << "  thenVal: " << thenVal << " -> thenCloned: " << (thenCloned ? "found" : "not found") << "\n";
        llvm::errs() << "  elseVal: " << elseVal << " -> elseCloned: " << (elseCloned ? "found" : "not found") << "\n";
        
        // Debug: print what we have
        llvm::errs() << "  Then map contents:\n";
        for (auto &mapping : thenMap.getValueMap()) {
          llvm::errs() << "    " << mapping.first << " -> " << mapping.second << "\n";
        }
        llvm::errs() << "  Else map contents:\n";
        for (auto &mapping : elseMap.getValueMap()) {
          llvm::errs() << "    " << mapping.first << " -> " << mapping.second << "\n";
        }
        
        return failure();
      }

      // Create select operation using the cloned values
      Value selectResult = rewriter.create<arith::SelectOp>(
          ifOp.getLoc(), condition, thenCloned, elseCloned);
      
      newResults.push_back(selectResult);
      llvm::errs() << "IfConversion: Created select for result type: " << result.getType() << "\n";
    }

    // Step 6: Replace the if operation with the new results
    rewriter.replaceOp(ifOp, newResults);
    
    llvm::errs() << "IfConversion: Successfully converted divergent branch to predicated operations\n";
    return success();
  }

private:
  bool isTriviallySpeculatable(Operation *op) const {
    // Check if operation is safe to speculate
    if (isa<arith::ConstantOp>(op)) return true;
    
    // Integer arithmetic operations
    if (isa<arith::AddIOp>(op)) return true;
    if (isa<arith::SubIOp>(op)) return true;
    if (isa<arith::MulIOp>(op)) return true;
    if (isa<arith::CmpIOp>(op)) return true;
    
    // Floating-point arithmetic operations
    if (isa<arith::AddFOp>(op)) return true;
    if (isa<arith::SubFOp>(op)) return true;
    if (isa<arith::MulFOp>(op)) return true;
    if (isa<arith::CmpFOp>(op)) return true;
    
    // Other safe operations
    if (isa<arith::SelectOp>(op)) return true;
    
    // For now, be conservative - only allow basic arithmetic operations
    // Later we can add MemoryEffects analysis for load/store operations
    return false;
  }

  bool hasNonSpeculatableOperations(scf::IfOp ifOp) const {
    // Check then region
    if (ifOp.getThenRegion().hasOneBlock()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      for (Operation &op : thenBlock.without_terminator()) {
        if (!isTriviallySpeculatable(&op)) {
          llvm::errs() << "IfConversion: Non-speculatable operation in then branch: " << op.getName() << "\n";
          return true;
        }
      }
    }
    
    // Check else region
    if (ifOp.getElseRegion().hasOneBlock()) {
      Block &elseBlock = ifOp.getElseRegion().front();
      for (Operation &op : elseBlock.without_terminator()) {
        if (!isTriviallySpeculatable(&op)) {
          llvm::errs() << "IfConversion: Non-speculatable operation in else branch: " << op.getName() << "\n";
          return true;
        }
      }
    }
    
    return false;
  }
};

struct IfConversionPass : public PassWrapper<IfConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfConversionPass)

  StringRef getArgument() const override { return "if-conversion"; }
  StringRef getDescription() const override { return "Convert divergent branches to predicated operations"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect, func::FuncDialect>();
    target.addIllegalOp<scf::IfOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<IfConversionPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createIfConversionPass() {
  return std::make_unique<IfConversionPass>();
}

namespace mlir {
namespace standalone {
void registerIfConversionPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> {
        return createIfConversionPass();
      });
}
} // namespace standalone
} // namespace mlir 
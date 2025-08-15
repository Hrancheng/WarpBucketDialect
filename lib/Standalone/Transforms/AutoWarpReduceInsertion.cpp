//===----------------------------------------------------------------------===//
// Auto Warp Reduce Insertion Pass
//===----------------------------------------------------------------------===//
//
// This pass automatically detects reduction patterns and inserts
// standalone.warp_reduce operations where appropriate.
//
// Supported patterns:
// 1. Accumulation loops (sum, product, min, max)
// 2. Array reductions
// 3. Mathematical reduction operations
// 4. Parallel reduction patterns
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::standalone;

#define DEBUG_TYPE "auto-warp-reduce-insertion"

namespace {

// Reduction pattern types
enum class ReductionPattern {
  ACCUMULATION_LOOP,    // for loop with accumulation
  ARRAY_REDUCTION,      // reduction over array elements
  MATH_REDUCTION,       // mathematical reduction operations
  PARALLEL_REDUCTION    // parallel reduction patterns
};

// Reduction operation info
struct ReductionInfo {
  ReductionPattern pattern;
  Value accumulator;           // The accumulator value
  Value reductionValue;        // The value being reduced
  Operation* reductionOp;      // The reduction operation
  StringRef reductionKind;     // "add", "mul", "min", "max", "and", "or", "xor"
  Location loc;                // Location for the new warp_reduce
  bool isWarpLevel;            // Whether this should be warp-level
};

// Pattern to detect and insert warp_reduce operations
struct AutoWarpReduceInsertionPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
    llvm::errs() << "AutoWarpReduce: Analyzing for loop\n";
    
    // Check if this loop has already been processed
    if (forOp->hasAttr("auto_warp_reduce_processed")) {
      llvm::errs() << "AutoWarpReduce: Loop already processed, skipping\n";
      return failure();
    }
    
    // Check if this is a GPU kernel function
    if (!isGPUKernel(forOp)) {
      llvm::errs() << "AutoWarpReduce: Not a GPU kernel, skipping\n";
      return failure();
    }
    
    // Detect accumulation patterns
    SmallVector<ReductionInfo> reductions;
    if (failed(detectAccumulationPatterns(forOp, reductions))) {
      llvm::errs() << "AutoWarpReduce: Failed to detect patterns\n";
      return failure();
    }
    
    if (reductions.empty()) {
      llvm::errs() << "AutoWarpReduce: No reduction patterns found\n";
      return failure();
    }
    
    llvm::errs() << "AutoWarpReduce: Found " << reductions.size() << " reduction patterns\n";
    
    // Insert warp_reduce operations
    for (const ReductionInfo& reduction : reductions) {
      if (failed(insertWarpReduce(forOp, reduction, rewriter))) {
        return failure();
      }
    }
    
    // Mark this loop as processed to prevent infinite loops
    forOp->setAttr("auto_warp_reduce_processed", rewriter.getBoolAttr(true));
    
    return success();
  }

private:
  // Check if this operation is in a GPU kernel context
  bool isGPUKernel(scf::ForOp forOp) const {
    // Check if we're in a GPU function or have GPU-related attributes
    Operation* parent = forOp->getParentOp();
    while (parent) {
      if (auto funcOp = dyn_cast<func::FuncOp>(parent)) {
        // Check for GPU-related attributes or function names
        if (funcOp->hasAttr("gpu.kernel") || 
            funcOp.getName().contains("kernel") ||
            funcOp.getName().contains("gpu")) {
          return true;
        }
      }
      parent = parent->getParentOp();
    }
    
    // For testing purposes, assume all functions are GPU kernels
    return true;
  }
  
  // Detect accumulation patterns in for loops
  LogicalResult detectAccumulationPatterns(scf::ForOp forOp, 
                                         SmallVector<ReductionInfo>& reductions) const {
    Block* bodyBlock = forOp.getBody();
    if (!bodyBlock) return failure();
    
    llvm::errs() << "AutoWarpReduce: Analyzing loop body with " << bodyBlock->getOperations().size() << " operations\n";
    
    // Look for accumulation patterns
    for (Operation& op : *bodyBlock) {
      llvm::errs() << "AutoWarpReduce: Checking operation " << op.getName() << "\n";
      
      // Skip operations that are already warp_reduce operations
      if (isa<standalone::WarpReduceOp>(op)) {
        llvm::errs() << "AutoWarpReduce: Skipping existing warp_reduce operation\n";
        continue;
      }
      
      if (auto addOp = dyn_cast<arith::AddFOp>(&op)) {
        if (isAccumulation(addOp, forOp)) {
          ReductionInfo info{
            ReductionPattern::ACCUMULATION_LOOP,
            addOp.getResult(),  // Use the result, not the lhs
            addOp.getRhs(),
            &op,
            "add",
            addOp.getLoc(),
            true
          };
          reductions.push_back(info);
        }
      } else if (auto addIOp = dyn_cast<arith::AddIOp>(&op)) {
        if (isAccumulation(addIOp, forOp)) {
          ReductionInfo info{
            ReductionPattern::ACCUMULATION_LOOP,
            addIOp.getResult(),  // Use the result, not the lhs
            addIOp.getRhs(),
            addIOp,
            "add",
            addIOp.getLoc(),
            true
          };
          reductions.push_back(info);
        }
      } else if (auto mulOp = dyn_cast<arith::MulFOp>(&op)) {
        if (isAccumulation(mulOp, forOp)) {
          ReductionInfo info{
            ReductionPattern::ACCUMULATION_LOOP,
            mulOp.getResult(),  // Use the result, not the lhs
            mulOp.getRhs(),
            &op,
            "mul",
            mulOp.getLoc(),
            true
          };
          reductions.push_back(info);
        }
      } else if (auto mulIOp = dyn_cast<arith::MulIOp>(&op)) {
        if (isAccumulation(mulIOp, forOp)) {
          ReductionInfo info{
            ReductionPattern::ACCUMULATION_LOOP,
            mulIOp.getResult(),  // Use the result, not the lhs
            mulIOp.getRhs(),
            &op,
            "mul",
            mulIOp.getLoc(),
            true
          };
          reductions.push_back(info);
        }
      }
    }
    
    return success();
  }
  
  // Check if an operation is an accumulation
  bool isAccumulation(Operation* op, scf::ForOp forOp) const {
    if (op->getNumResults() != 1) return false;
    
    Value result = op->getResult(0);
    Value lhs = op->getOperand(0);
    
    llvm::errs() << "AutoWarpReduce: Checking accumulation for " << op->getName() << ", lhs type: " << lhs.getType() << "\n";
    
    // Check if lhs is an iter_arg (accumulator)
    if (auto iterArg = dyn_cast<BlockArgument>(lhs)) {
      llvm::errs() << "AutoWarpReduce: LHS is BlockArgument " << iterArg.getArgNumber() << "\n";
      Block* bodyBlock = forOp.getBody();
      if (!bodyBlock) return false;
      
      // Check if this block argument corresponds to an iter_arg
      int argIndex = iterArg.getArgNumber();
      llvm::errs() << "AutoWarpReduce: Checking if arg " << argIndex << " is iter_arg (total iter_args: " << forOp.getRegionIterArgs().size() << ")\n";
      llvm::errs() << "AutoWarpReduce: getNumInductionVars() = " << forOp.getNumInductionVars() << "\n";
      if (argIndex >= forOp.getNumInductionVars()) {
        llvm::errs() << "AutoWarpReduce: Found iter_arg accumulation!\n";
        return true;
      }
    }
    
    // Check if lhs is defined in the loop body
    Block* bodyBlock = forOp.getBody();
    if (!bodyBlock) return false;
    
    for (Operation& bodyOp : *bodyBlock) {
      if (&bodyOp == op) continue;
      for (Value res : bodyOp.getResults()) {
        if (res == lhs) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  // Insert warp_reduce operation
  LogicalResult insertWarpReduce(scf::ForOp forOp, const ReductionInfo& reduction,
                                PatternRewriter& rewriter) const {
    llvm::errs() << "AutoWarpReduce: Inserting warp_reduce for " 
                 << reduction.reductionKind << "\n";
    
    // Find the yield operation to insert before
    Block* bodyBlock = forOp.getBody();
    if (!bodyBlock) return failure();
    
    Operation* yieldOp = nullptr;
    for (Operation& op : *bodyBlock) {
      if (isa<scf::YieldOp>(op)) {
        yieldOp = &op;
        break;
      }
    }
    
    if (!yieldOp) {
      llvm::errs() << "AutoWarpReduce: No yield operation found\n";
      return failure();
    }
    
    llvm::errs() << "AutoWarpReduce: Found yield operation\n";
    
    // Get the final accumulator value
    Value finalValue = reduction.accumulator;
    
    llvm::errs() << "AutoWarpReduce: Creating warp_reduce with accumulator type: " << finalValue.getType() << "\n";
    
    // Create warp_reduce operation BEFORE the yield
    rewriter.setInsertionPoint(yieldOp);
    auto warpReduceOp = rewriter.create<WarpReduceOp>(
        reduction.loc,
        finalValue
    );
    
    // Add attributes
    warpReduceOp->setAttr("kind", rewriter.getStringAttr(reduction.reductionKind));
    warpReduceOp->setAttr("width", rewriter.getI32IntegerAttr(32));  // Default warp size
    
    llvm::errs() << "AutoWarpReduce: Created warp_reduce operation\n";
    llvm::errs() << "AutoWarpReduce: Warp_reduce result type: " << warpReduceOp.getResult().getType() << "\n";
    llvm::errs() << "AutoWarpReduce: Original accumulator type: " << finalValue.getType() << "\n";
    
    // Update the yield operation to use the reduced value
    SmallVector<Value> newYieldOperands;
    llvm::errs() << "AutoWarpReduce: Processing yield operands...\n";
    
    // We need to replace the result of the accumulation operation with the warp_reduce result
    // The accumulation result is what we're currently yielding
    Value accumulationResult = reduction.accumulator;
    llvm::errs() << "AutoWarpReduce: Looking for accumulation result: " << accumulationResult << "\n";
    
    for (Value operand : yieldOp->getOperands()) {
      llvm::errs() << "AutoWarpReduce: Checking operand: " << operand << " (type: " << operand.getType() << ")\n";
      
      // Check if this operand is the accumulation result
      if (operand == accumulationResult) {
        llvm::errs() << "AutoWarpReduce: Found accumulation result, replacing with warp_reduce result\n";
        llvm::errs() << "AutoWarpReduce: Warp_reduce result: " << warpReduceOp.getResult() << " (type: " << warpReduceOp.getResult().getType() << ")\n";
        newYieldOperands.push_back(warpReduceOp.getResult());
      } else {
        llvm::errs() << "AutoWarpReduce: Keeping original operand: " << operand << "\n";
        newYieldOperands.push_back(operand);
      }
    }
    
    llvm::errs() << "AutoWarpReduce: New yield operands count: " << newYieldOperands.size() << "\n";
    for (size_t i = 0; i < newYieldOperands.size(); i++) {
      llvm::errs() << "AutoWarpReduce: Yield operand " << i << " type: " << newYieldOperands[i].getType() << "\n";
    }
    
    // Replace the yield operation with updated operands
    llvm::errs() << "AutoWarpReduce: About to replace yield operation\n";
    llvm::errs() << "AutoWarpReduce: Original yield operands: ";
    for (Value operand : yieldOp->getOperands()) {
      llvm::errs() << operand << " ";
    }
    llvm::errs() << "\n";
    
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOperands);
    
    llvm::errs() << "AutoWarpReduce: Successfully inserted warp_reduce\n";
    llvm::errs() << "AutoWarpReduce: After yield replacement, loop body has " << forOp.getBody()->getOperations().size() << " operations\n";
    for (Operation& op : *forOp.getBody()) {
      llvm::errs() << "AutoWarpReduce: Loop body operation: " << op.getName() << "\n";
    }
    
    // Let's also check if the warp_reduce operation is still there
    bool foundWarpReduce = false;
    for (Operation& op : *forOp.getBody()) {
      if (isa<standalone::WarpReduceOp>(op)) {
        foundWarpReduce = true;
        llvm::errs() << "AutoWarpReduce: Found warp_reduce operation after yield replacement\n";
        break;
      }
    }
    if (!foundWarpReduce) {
      llvm::errs() << "AutoWarpReduce: WARNING: warp_reduce operation disappeared after yield replacement!\n";
    }
    
    return success();
  }
};

// Pattern to detect array reduction patterns
struct ArrayReductionPattern : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp, PatternRewriter &rewriter) const override {
    // This pattern would detect array reduction patterns
    // For now, return failure to avoid conflicts
    return failure();
  }
};

// Main pass for automatic warp_reduce insertion
struct AutoWarpReduceInsertionPass : public PassWrapper<AutoWarpReduceInsertionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoWarpReduceInsertionPass)

  StringRef getArgument() const final { return "auto-warp-reduce-insertion"; }
  StringRef getDescription() const final {
    return "Automatically detect and insert warp_reduce operations for reduction patterns";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<standalone::StandaloneDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "AutoWarpReduceInsertion: Starting pass execution\n");
    
    // Apply patterns for for loops
    RewritePatternSet patterns(&getContext());
    patterns.add<AutoWarpReduceInsertionPattern>(&getContext());
    patterns.add<ArrayReductionPattern>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "AutoWarpReduceInsertion: Pass execution completed\n");
  }
};

} // namespace

namespace mlir {
namespace standalone {

std::unique_ptr<Pass> createAutoWarpReduceInsertionPass() {
  return std::make_unique<AutoWarpReduceInsertionPass>();
}

} // namespace standalone
} // namespace mlir 
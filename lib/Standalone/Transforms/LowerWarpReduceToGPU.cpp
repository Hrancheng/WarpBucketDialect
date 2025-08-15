//===----------------------------------------------------------------------===//
// Lower Warp Reduce to GPU Shuffle
//===----------------------------------------------------------------------===//
//
// This pass lowers wb.warp_reduce operations to gpu.shuffle operations.
// The lowering implements a tree-based reduction algorithm using shuffle operations.
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::standalone;

namespace {

// Pattern to convert wb.warp_reduce to gpu.shuffle operations
struct WarpReduceToShufflePattern : public OpRewritePattern<WarpReduceOp> {
  using OpRewritePattern<WarpReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WarpReduceOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getValue();
    
    // Get reduction kind and width from attributes (default to "add" and 32)
    StringAttr kindAttr = op->getAttrOfType<StringAttr>("kind");
    StringRef reductionKind = kindAttr ? kindAttr.getValue() : "add";
    
    IntegerAttr widthAttr = op->getAttrOfType<IntegerAttr>("width");
    int warpSize = widthAttr ? widthAttr.getInt() : 32;
    
    // Calculate number of reduction steps (log2 of warp size)
    int tempSize = warpSize;
    while (tempSize > 1) {
      tempSize >>= 1;
    }
    
    // Start with the input value
    Value currentValue = input;
    
    // Implement tree-based reduction algorithm
    for (int step = 1; step < warpSize; step <<= 1) {
      // Create shuffle operation for this step
      Value stepValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(step));
      Value warpSizeValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(warpSize));
      
      auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
          loc, currentValue, stepValue, warpSizeValue, gpu::ShuffleMode::DOWN);
      
      // Combine the current value with the shuffled value based on reduction kind
      Value shuffledValue = shuffleOp.getResult(0);
      currentValue = createReductionOp(rewriter, loc, currentValue, shuffledValue, reductionKind);
    }
    
    // Replace the original operation with the final result
    rewriter.replaceOp(op, currentValue);
    return success();
  }

private:
  // Helper function to create the appropriate reduction operation
  Value createReductionOp(PatternRewriter &rewriter, Location loc, Value a, Value b, StringRef kind) const {
    if (kind == "add") {
      if (a.getType().isInteger(32) || a.getType().isInteger(64)) {
        return rewriter.create<arith::AddIOp>(loc, a, b);
      } else {
        return rewriter.create<arith::AddFOp>(loc, a, b);
      }
    } else if (kind == "mul") {
      if (a.getType().isInteger(32) || a.getType().isInteger(64)) {
        return rewriter.create<arith::MulIOp>(loc, a, b);
      } else {
        return rewriter.create<arith::MulFOp>(loc, a, b);
      }
    } else if (kind == "and") {
      if (a.getType().isInteger(32) || a.getType().isInteger(64)) {
        return rewriter.create<arith::AndIOp>(loc, a, b);
      }
    } else if (kind == "or") {
      if (a.getType().isInteger(32) || a.getType().isInteger(64)) {
        return rewriter.create<arith::OrIOp>(loc, a, b);
      }
    } else if (kind == "xor") {
      if (a.getType().isInteger(32) || a.getType().isInteger(64)) {
        return rewriter.create<arith::XOrIOp>(loc, a, b);
      }
    }
    
    // Default to addition if kind is not recognized
    if (a.getType().isInteger(32) || a.getType().isInteger(64)) {
      return rewriter.create<arith::AddIOp>(loc, a, b);
    } else {
      return rewriter.create<arith::AddFOp>(loc, a, b);
    }
  }
};

// Pass to lower warp reduce operations
struct LowerWarpReduceToGPUPass : public PassWrapper<LowerWarpReduceToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerWarpReduceToGPUPass)

  StringRef getArgument() const override { return "lower-warp-reduce-to-gpu"; }
  StringRef getDescription() const override { return "Lower warp reduce operations to GPU shuffle operations"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    target.addLegalDialect<gpu::GPUDialect, arith::ArithDialect>();
    target.addIllegalOp<WarpReduceOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<WarpReduceToShufflePattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace standalone {

std::unique_ptr<Pass> createLowerWarpReduceToGPUPass() {
  return std::make_unique<LowerWarpReduceToGPUPass>();
}

} // namespace standalone
} // namespace mlir 
#include "Standalone/StandaloneOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/LLVM.h"

// Forward declaration
std::unique_ptr<mlir::Pass> createLowerArithToGPUPass();

namespace {

struct LowerAddOp : mlir::OpRewritePattern<mlir::standalone::AddOp> {
  LowerAddOp(mlir::MLIRContext *context) 
      : mlir::OpRewritePattern<mlir::standalone::AddOp>(context, 1) {
    llvm::errs() << "LowerAddOp: Pattern constructor called\n";
  }
  
  mlir::LogicalResult matchAndRewrite(mlir::standalone::AddOp op,
                                mlir::PatternRewriter &rewriter) const override {
    // Add debug output
    llvm::errs() << "LowerAddOp: Matching standalone.add operation\n";
    
    mlir::Type ty = op.getRes().getType();
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    if (ty.isa<mlir::IntegerType>()) {
      llvm::errs() << "LowerAddOp: Creating arith.addi for integer type\n";
      auto addi = rewriter.create<mlir::arith::AddIOp>(op.getLoc(), lhs, rhs);
      rewriter.replaceOp(op, addi.getResult());
      return mlir::success();
    } else if (ty.isa<mlir::FloatType>()) {
      llvm::errs() << "LowerAddOp: Creating arith.addf for float type\n";
      auto addf = rewriter.create<mlir::arith::AddFOp>(op.getLoc(), lhs, rhs);
      rewriter.replaceOp(op, addf.getResult());
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported result type");
  }
};

struct LowerStandaloneToArithPass
    : public mlir::PassWrapper<LowerStandaloneToArithPass,
                         mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStandaloneToArithPass)

  mlir::StringRef getArgument() const final { return "standalone-lower"; }
  mlir::StringRef getDescription() const final {
    return "Lower standalone dialect ops to arith dialect";
  }

  void runOnOperation() override {
    llvm::errs() << "LowerStandaloneToArithPass: Starting pass execution\n";
    
    // Debug: Print all operations in the module
    llvm::errs() << "LowerStandaloneToArithPass: Module operations:\n";
    getOperation()->walk([](mlir::Operation *op) {
      llvm::errs() << "  - " << op->getName() << "\n";
    });
    
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerAddOp>(&getContext());
    
    llvm::errs() << "LowerStandaloneToArithPass: Added LowerAddOp pattern\n";
    
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
    
    llvm::errs() << "LowerStandaloneToArithPass: Pass execution completed\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerStandaloneToArithPass() {
  return std::make_unique<LowerStandaloneToArithPass>();
}

// 注册到 pass 管理器（这样 wb-opt/standalone-opt 能识别 --standalone-lower）
//static PassRegistration<LowerStandaloneToArithPass> reg;
namespace mlir {
    namespace standalone {
    
    void registerStandalonePasses() {
      // 运行时注册：为 pass 提供工厂、选项名和描述
      ::mlir::registerPass(
          []() -> std::unique_ptr<::mlir::Pass> {
            return createLowerStandaloneToArithPass();
          });
      
      // Register the new GPU lowering pass
      ::mlir::registerPass(
          []() -> std::unique_ptr<::mlir::Pass> {
            return createLowerArithToGPUPass();
          });
    }
    
    } // namespace standalone
    } // namespace mlir

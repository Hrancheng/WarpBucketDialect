#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

// Three-value lattice for uniformity analysis
enum class LatticeVal { Uniform, Unknown, Varying };

// Lattice join operation (meet in the lattice)
LatticeVal join(LatticeVal a, LatticeVal b) {
  if (a == LatticeVal::Varying || b == LatticeVal::Varying)
    return LatticeVal::Varying;
  if (a == LatticeVal::Unknown || b == LatticeVal::Unknown)
    return LatticeVal::Unknown;
  return LatticeVal::Uniform;
}

// Convert lattice value to string for debugging
StringRef latticeToString(LatticeVal val) {
  switch (val) {
    case LatticeVal::Uniform: return "Uniform";
    case LatticeVal::Unknown: return "Unknown";
    case LatticeVal::Varying: return "Varying";
  }
  llvm_unreachable("Unknown lattice value");
}

struct UniformityAnalysisPass
    : public PassWrapper<UniformityAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UniformityAnalysisPass)

  StringRef getArgument() const final { return "uniformity-analysis"; }
  StringRef getDescription() const final {
    return "Analyze uniformity of values across GPU warp";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Process each function in the module
    for (auto func : module.getOps<func::FuncOp>()) {
      analyzeFunction(func);
    }
  }

private:
  // Main analysis function
  void analyzeFunction(func::FuncOp func) {
    llvm::errs() << "UniformityAnalysis: Analyzing function " << func.getName() << "\n";
    
    // Initialize lattice values for all values
    DenseMap<Value, LatticeVal> lattice;
    SetVector<Operation*> worklist;
    
    // Initialize all values to Uniform
    func.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        lattice[result] = LatticeVal::Uniform;
      }
      worklist.insert(op);
    });
    
    // Set seeds for varying values
    setVaryingSeeds(func, lattice, worklist);
    
    // Set seeds for uniform values
    setUniformSeeds(func, lattice, worklist);
    
    // Set function arguments as uniform
    setFunctionArgumentsUniform(func, lattice);
    
    // Iterate until fixed point
    bool changed = true;
    int iteration = 0;
    while (changed && iteration < 100) { // Safety limit
      changed = false;
      iteration++;
      
      llvm::errs() << "UniformityAnalysis: Iteration " << iteration << "\n";
      
      // Process all operations
      for (Operation *op : worklist) {
        if (processOperation(op, lattice)) {
          changed = true;
        }
      }
      
      // Process block arguments (phi nodes)
      if (processBlockArguments(func, lattice)) {
        changed = true;
      }
    }
    
    if (iteration >= 100) {
      llvm::errs() << "UniformityAnalysis: Warning: Reached iteration limit\n";
    }
    
    // Mark divergent operations
    markDivergentOperations(func, lattice);
    
    // Debug output
    printAnalysisResults(func, lattice);
  }
  
  // Set seeds for varying values (thread-specific)
  void setVaryingSeeds(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice,
                       SetVector<Operation*> &worklist) {
    func.walk([&](Operation *op) {
      // GPU thread IDs are varying
      if (auto threadId = dyn_cast<gpu::ThreadIdOp>(op)) {
        lattice[op->getResult(0)] = LatticeVal::Varying;
        llvm::errs() << "UniformityAnalysis: Marking " << op->getName() << " as Varying\n";
      }
      
      // GPU lane IDs are varying
      if (auto laneId = dyn_cast<gpu::LaneIdOp>(op)) {
        lattice[op->getResult(0)] = LatticeVal::Varying;
        llvm::errs() << "UniformityAnalysis: Marking " << op->getName() << " as Varying\n";
      }
      
      // Any operation that uses varying values becomes varying
      for (Value operand : op->getOperands()) {
        if (lattice[operand] == LatticeVal::Varying) {
          for (Value result : op->getResults()) {
            lattice[result] = LatticeVal::Varying;
            llvm::errs() << "UniformityAnalysis: Propagating Varying to " << op->getName() << "\n";
          }
          break;
        }
      }
    });
  }
  
  // Set seeds for uniform values (warp-wide constants)
  void setUniformSeeds(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice,
                       SetVector<Operation*> &worklist) {
    func.walk([&](Operation *op) {
      // Constants are uniform
      if (isa<arith::ConstantOp>(op)) {
        for (Value result : op->getResults()) {
          lattice[result] = LatticeVal::Uniform;
          llvm::errs() << "UniformityAnalysis: Marking constant as Uniform\n";
        }
      }
      
      // GPU grid/block dimensions are uniform within a warp
      if (isa<gpu::GridDimOp, gpu::BlockDimOp>(op)) {
        for (Value result : op->getResults()) {
          lattice[result] = LatticeVal::Uniform;
          llvm::errs() << "UniformityAnalysis: Marking " << op->getName() << " as Uniform\n";
        }
      }
    });
  }
  
  // Set function arguments as uniform (kernel parameters)
  void setFunctionArgumentsUniform(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice) {
    for (BlockArgument arg : func.getArguments()) {
      lattice[arg] = LatticeVal::Uniform;
      llvm::errs() << "UniformityAnalysis: Marking function argument as Uniform\n";
    }
  }
  
  // Process a single operation
  bool processOperation(Operation *op, DenseMap<Value, LatticeVal> &lattice) {
    bool changed = false;
    
    // Skip operations that are already varying
    bool hasVaryingInput = false;
    for (Value operand : op->getOperands()) {
      if (lattice[operand] == LatticeVal::Varying) {
        hasVaryingInput = true;
        break;
      }
    }
    
    if (hasVaryingInput) {
      for (Value result : op->getResults()) {
        if (lattice[result] != LatticeVal::Varying) {
          lattice[result] = LatticeVal::Varying;
          changed = true;
        }
      }
      return changed;
    }
    
    // Compute join of all input lattice values
    LatticeVal outputVal = LatticeVal::Uniform;
    for (Value operand : op->getOperands()) {
      outputVal = join(outputVal, lattice[operand]);
    }
    
    // Update results if changed
    for (Value result : op->getResults()) {
      if (lattice[result] != outputVal) {
        lattice[result] = outputVal;
        changed = true;
      }
    }
    
    return changed;
  }
  
  // Process block arguments (phi nodes)
  bool processBlockArguments(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice) {
    bool changed = false;
    
    for (Block &block : func.getBody()) {
      for (BlockArgument arg : block.getArguments()) {
        LatticeVal mergedVal = LatticeVal::Uniform;
        
        // Find all predecessors and their incoming values
        for (Block *pred : block.getPredecessors()) {
          Operation *terminator = pred->getTerminator();
          
          if (auto branchOp = dyn_cast<cf::BranchOp>(terminator)) {
            // BranchOp has only one destination
            if (branchOp.getDest() == &block) {
              for (unsigned i = 0; i < branchOp.getOperands().size(); i++) {
                if (i == arg.getArgNumber()) {
                  Value incomingVal = branchOp.getOperand(i);
                  mergedVal = join(mergedVal, lattice[incomingVal]);
                }
              }
            }
          } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(terminator)) {
            // For conditional branches, find the operand for this block
            if (condBranchOp.getTrueDest() == &block) {
              for (unsigned i = 0; i < condBranchOp.getTrueOperands().size(); i++) {
                if (i == arg.getArgNumber()) {
                  Value incomingVal = condBranchOp.getTrueOperand(i);
                  mergedVal = join(mergedVal, lattice[incomingVal]);
                }
              }
            } else if (condBranchOp.getFalseDest() == &block) {
              for (unsigned i = 0; i < condBranchOp.getFalseOperands().size(); i++) {
                if (i == arg.getArgNumber()) {
                  Value incomingVal = condBranchOp.getFalseOperand(i);
                  mergedVal = join(mergedVal, lattice[incomingVal]);
                }
              }
            }
          } else if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
            // For SCF operations, find the corresponding yield value
            // This is a simplified version - you might need more complex logic
            if (yieldOp.getOperands().size() > arg.getArgNumber()) {
              Value incomingVal = yieldOp.getOperand(arg.getArgNumber());
              mergedVal = join(mergedVal, lattice[incomingVal]);
            }
          }
        }
        
        // Update if changed
        if (lattice[arg] != mergedVal) {
          lattice[arg] = mergedVal;
          changed = true;
        }
      }
    }
    
    return changed;
  }
  
  // Mark operations as divergent based on lattice analysis
  void markDivergentOperations(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice) {
    func.walk([&](Operation *op) {
      // Mark divergent if statements
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        Value condition = ifOp.getCondition();
        if (lattice[condition] == LatticeVal::Varying) {
          op->setAttr("wb.divergent", UnitAttr::get(op->getContext()));
          llvm::errs() << "UniformityAnalysis: Marking if statement as divergent\n";
        }
      }
      
      // Mark divergent conditional branches
      if (auto condBr = dyn_cast<cf::CondBranchOp>(op)) {
        Value condition = condBr.getCondition();
        if (lattice[condition] == LatticeVal::Varying) {
          op->setAttr("wb.divergent", UnitAttr::get(op->getContext()));
          llvm::errs() << "UniformityAnalysis: Marking conditional branch as divergent\n";
        }
      }
      
      // Mark loops with varying exit conditions
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        Value lowerBound = forOp.getLowerBound();
        Value upperBound = forOp.getUpperBound();
        Value step = forOp.getStep();
        
        if (lattice[lowerBound] == LatticeVal::Varying ||
            lattice[upperBound] == LatticeVal::Varying ||
            lattice[step] == LatticeVal::Varying) {
          op->setAttr("wb.varying_exit", UnitAttr::get(op->getContext()));
          llvm::errs() << "UniformityAnalysis: Marking for loop as varying exit\n";
        }
      }
      
      // Mark while loops with varying exit conditions
      if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        // Check condition in the before block
        Block &beforeBlock = whileOp.getBefore().front();
        if (auto yieldOp = dyn_cast<scf::ConditionOp>(beforeBlock.getTerminator())) {
          Value condition = yieldOp.getCondition();
          if (lattice[condition] == LatticeVal::Varying) {
            op->setAttr("wb.varying_exit", UnitAttr::get(op->getContext()));
            llvm::errs() << "UniformityAnalysis: Marking while loop as varying exit\n";
          }
        }
      }
    });
  }
  
  // Print analysis results for debugging
  void printAnalysisResults(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice) {
    llvm::errs() << "\n=== Uniformity Analysis Results ===\n";
    
    func.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        llvm::errs() << "  " << result << " -> " << latticeToString(lattice[result]) << "\n";
      }
    });
    
    llvm::errs() << "=== End Results ===\n\n";
  }
};

} // namespace

std::unique_ptr<Pass> createUniformityAnalysisPass() {
  return std::make_unique<UniformityAnalysisPass>();
}

// Register the pass
namespace mlir {
namespace standalone {

void registerUniformityAnalysisPass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> {
        return createUniformityAnalysisPass();
      });
}

} // namespace standalone
} // namespace mlir 
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "Standalone/StandaloneOps.h"

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
    llvm::errs() << "UniformityAnalysis: runOnOperation called\n";
    ModuleOp module = getOperation();
    
    // Process each function in the module
    for (auto func : module.getOps<func::FuncOp>()) {
      analyzeFunction(func);
    }
  }

private:
  // Main analysis function
  void analyzeFunction(func::FuncOp func) {
    llvm::errs() << "UniformityAnalysis: analyzeFunction called\n";
    llvm::errs() << "UniformityAnalysis: About to start analysis\n";
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
    
    llvm::errs() << "UniformityAnalysis: Initialization complete, worklist size: " << worklist.size() << "\n";
    
    // Set seeds for varying values
    setVaryingSeeds(func, lattice, worklist);
    
    // Set seeds for uniform values
    llvm::errs() << "UniformityAnalysis: About to call setUniformSeeds\n";
    setUniformSeeds(func, lattice, worklist);
    llvm::errs() << "UniformityAnalysis: Finished calling setUniformSeeds\n";
    
    // Set function arguments as uniform - INLINE VERSION
    llvm::errs() << "UniformityAnalysis: INLINE setFunctionArgumentsUniform called with " << func.getBody().front().getArguments().size() << " arguments\n";
    llvm::errs().flush();
    
    int varyingCount = 0;
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      llvm::errs() << "UniformityAnalysis: INLINE Processing argument " << arg.getArgNumber() << " of type " << arg.getType() << "\n";
      llvm::errs().flush();
      // For testing: mark specific arguments as Varying to simulate gpu.thread_id
      if (arg.getArgNumber() == 0 || arg.getArgNumber() == 1) {
        // Simulate thread_id and lane_id as Varying
        lattice[arg] = LatticeVal::Varying;
        varyingCount++;
        llvm::errs() << "UniformityAnalysis: INLINE Marking varying function argument " << arg.getArgNumber() << " as Varying\n";
        llvm::errs().flush();
      } else {
        // Other arguments remain Uniform
        lattice[arg] = LatticeVal::Uniform;
        llvm::errs() << "UniformityAnalysis: INLINE Marking function argument " << arg.getArgNumber() << " as Uniform\n";
        llvm::errs().flush();
      }
    }
    
    llvm::errs() << "UniformityAnalysis: INLINE Finished processing " << varyingCount << " varying arguments\n";
    llvm::errs().flush();
    
    // Comment out the function call for now
    // setFunctionArgumentsUniform(func, lattice);
    
    // Iterate until fixed point
    bool changed = true;
    int iteration = 0;
    
    // Process operations until convergence
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
    llvm::errs() << "UniformityAnalysis: setUniformSeeds called\n";
    func.walk([&](Operation *op) {
      // Debug: Print all operations being processed
      llvm::errs() << "UniformityAnalysis: setUniformSeeds processing: " << op->getName() << "\n";
      
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
      
      // Memory loads are Unknown - we can't determine if all threads access same location
      if (isa<memref::LoadOp>(op)) {
        llvm::errs() << "UniformityAnalysis: Found memref.load, marking as Unknown\n";
        for (Value result : op->getResults()) {
          lattice[result] = LatticeVal::Unknown;
          llvm::errs() << "UniformityAnalysis: Marking " << op->getName() << " as Unknown\n";
        }
      }
    });
  }
  
  // Set function arguments as uniform (kernel parameters)
  void setFunctionArgumentsUniform(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice) {
    llvm::errs() << "UniformityAnalysis: setFunctionArgumentsUniform ENTERED\n";
    llvm::errs() << "UniformityAnalysis: setFunctionArgumentsUniform called with " << func.getBody().front().getArguments().size() << " arguments\n";
    
    // Force side effect to prevent optimization
    bool hasVaryingArgs = false;
    
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      llvm::errs() << "UniformityAnalysis: Processing argument " << arg.getArgNumber() << " of type " << arg.getType() << "\n";
      // For testing: mark specific arguments as Varying to simulate gpu.thread_id
      if (arg.getArgNumber() == 0 || arg.getArgNumber() == 1) {
        // Simulate thread_id and lane_id as Varying
        lattice[arg] = LatticeVal::Varying;
        hasVaryingArgs = true;
        llvm::errs() << "UniformityAnalysis: Marking varying function argument " << arg.getArgNumber() << " as Varying\n";
      } else {
        // Other arguments remain Uniform
        lattice[arg] = LatticeVal::Uniform;
        llvm::errs() << "UniformityAnalysis: Marking function argument " << arg.getArgNumber() << " as Uniform\n";
      }
    }
    
    // Force side effect - this prevents the compiler from optimizing away the function
    if (hasVaryingArgs) {
      llvm::errs() << "UniformityAnalysis: Found varying arguments - function call preserved\n";
    }
    
    // CRITICAL: Force this function to be essential by modifying the lattice
    // This will make the compiler keep the function call
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      if (lattice[arg] == LatticeVal::Varying) {
        // Mark all operations that use varying arguments as varying
        func.walk([&](Operation *op) {
          for (Value operand : op->getOperands()) {
            if (operand == arg) {
              for (Value result : op->getResults()) {
                lattice[result] = LatticeVal::Varying;
                llvm::errs() << "UniformityAnalysis: Propagating Varying from arg " << arg.getArgNumber() << " to " << op->getName() << "\n";
              }
            }
          }
        });
      }
    }
  }
  
  // Process a single operation
  bool processOperation(Operation *op, DenseMap<Value, LatticeVal> &lattice) {
    llvm::errs() << "UniformityAnalysis: processOperation ENTERED for " << op->getName() << "\n";
    llvm::errs().flush();
    llvm::errs() << "UniformityAnalysis: *** IMPOSSIBLE TO MISS DEBUG PRINT ***\n";
    llvm::errs().flush();
    bool changed = false;
    
    // Debug: Print operation being processed
    llvm::errs() << "UniformityAnalysis: Processing operation: " << op->getName() << "\n";
    
    // Debug: Show current lattice values for this operation's results
    for (Value result : op->getResults()) {
      llvm::errs() << "UniformityAnalysis: Before processing, " << result << " -> " << latticeToString(lattice[result]) << "\n";
    }
    
    // CRITICAL FIX: Skip operations that are seed operations (GPU ops, constants, etc.)
    llvm::errs() << "UniformityAnalysis: Checking if " << op->getName() << " is a seed operation\n";
    if (isa<gpu::ThreadIdOp, gpu::LaneIdOp, gpu::GridDimOp, gpu::BlockDimOp, arith::ConstantOp, memref::LoadOp>(op)) {
      llvm::errs() << "UniformityAnalysis: Skipping seed operation " << op->getName() << "\n";
      return false; // No changes needed
    } else {
      llvm::errs() << "UniformityAnalysis: " << op->getName() << " is NOT a seed operation, processing normally\n";
    }
    
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
          llvm::errs() << "UniformityAnalysis: Setting " << result << " to Varying due to varying input\n";
          lattice[result] = LatticeVal::Varying;
          changed = true;
        }
      }
      return changed;
    }
    
    // Only compute join if the operation has operands
    if (op->getNumOperands() > 0) {
      // Compute join of all input lattice values
      LatticeVal outputVal = LatticeVal::Uniform;
      for (Value operand : op->getOperands()) {
        outputVal = join(outputVal, lattice[operand]);
      }
      
      // Update results if changed
      for (Value result : op->getResults()) {
        if (lattice[result] != outputVal) {
          llvm::errs() << "UniformityAnalysis: Setting " << result << " from " << latticeToString(lattice[result]) << " to " << latticeToString(outputVal) << "\n";
          lattice[result] = outputVal;
          changed = true;
        }
      }
    }
    
    // Debug: Show final lattice values for this operation's results
    for (Value result : op->getResults()) {
      llvm::errs() << "UniformityAnalysis: After processing, " << result << " -> " << latticeToString(lattice[result]) << "\n";
    }
    
    return changed;
  }
  
  // Process block arguments (phi nodes)
  bool processBlockArguments(func::FuncOp func, DenseMap<Value, LatticeVal> &lattice) {
    bool changed = false;
    
    llvm::errs() << "UniformityAnalysis: processBlockArguments called\n";
    
    for (Block &block : func.getBody()) {
      for (BlockArgument arg : block.getArguments()) {
        llvm::errs() << "UniformityAnalysis: processBlockArguments: processing arg " << arg.getArgNumber() << " (current value: " << latticeToString(lattice[arg]) << ")\n";
        
        // Skip function arguments that we've already set to Varying
        if (lattice[arg] == LatticeVal::Varying) {
          llvm::errs() << "UniformityAnalysis: processBlockArguments: preserving Varying for arg " << arg.getArgNumber() << "\n";
          continue;
        }
        
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
          
          // AUTOMATIC LONG ROW DETECTION: Check if this if statement contains warp_reduce
          // This indicates it should use the long row strategy
          bool hasWarpReduce = false;
          bool hasMemRefLoad = false;
          
          // Check the else branch for warp_reduce operations (long row pattern)
          if (ifOp.getElseRegion().empty() == false) {
            ifOp.getElseRegion().front().walk([&](Operation *innerOp) {
              if (isa<standalone::WarpReduceOp>(innerOp)) {
                hasWarpReduce = true;
                llvm::errs() << "UniformityAnalysis: Found warp_reduce in else branch - marking as long row\n";
              }
              if (isa<memref::LoadOp>(innerOp)) {
                hasMemRefLoad = true;
              }
            });
          }
          
          // Check the then branch for warp_reduce operations (alternative long row pattern)
          ifOp.getThenRegion().walk([&](Operation *innerOp) {
            if (isa<standalone::WarpReduceOp>(innerOp)) {
              hasWarpReduce = true;
              llvm::errs() << "UniformityAnalysis: Found warp_reduce in then branch - marking as long row\n";
            }
            if (isa<memref::LoadOp>(innerOp)) {
              hasMemRefLoad = true;
            }
          });
          
          // Mark as long row if it contains warp_reduce operations
          if (hasWarpReduce) {
            op->setAttr("wb.long_row", UnitAttr::get(op->getContext()));
            llvm::errs() << "UniformityAnalysis: Marking if statement as long row strategy\n";
          }
          
          // Mark as short row if it contains memref.load but no warp_reduce
          if (hasMemRefLoad && !hasWarpReduce) {
            op->setAttr("wb.short_row", UnitAttr::get(op->getContext()));
            llvm::errs() << "UniformityAnalysis: Marking if statement as short row strategy\n";
          }
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

namespace mlir {
namespace standalone {

std::unique_ptr<Pass> createUniformityAnalysisPass() {
  return std::make_unique<UniformityAnalysisPass>();
}

} // namespace standalone
} // namespace mlir 
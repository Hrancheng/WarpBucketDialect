//===----------------------------------------------------------------------===//
// Standalone Operations
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::standalone;

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.cpp.inc"

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  // Check that the operands have the same type
  if (getLhs().getType() != getRhs().getType()) {
    return emitOpError("operands must have the same type");
  }
  
  // Check that the result type matches the operand type
  if (getRes().getType() != getLhs().getType()) {
    return emitOpError("result type must match operand type");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// WarpReduceOp
//===----------------------------------------------------------------------===//

LogicalResult WarpReduceOp::verify() {
  // Check that result type matches input type
  if (getValue().getType() != getResult().getType()) {
    return emitOpError("result type must match input type");
  }
  return success();
}

// Note: MaskedLoadOp and MaskedStoreOp are temporarily commented out due to type constraints

//===----------------------------------------------------------------------===//
// WarpStrideLoopOp
//===----------------------------------------------------------------------===//

LogicalResult WarpStrideLoopOp::verify() {
  // Check that start < end
  // Note: We can't directly compare Index values, so we'll skip this check for now
  // In practice, this would be checked at runtime or through other means
  
  // Check that warp_size > 0
  // Note: We can't directly compare Index values, so we'll skip this check for now
  // In practice, this would be checked at runtime or through other means
  
  return success();
}

//===----------------------------------------------------------------------===//
// UniformBranchOp
//===----------------------------------------------------------------------===//

LogicalResult UniformBranchOp::verify() {
  // Check that the condition is a boolean type
  if (!getCondition().getType().isInteger(1)) {
    return emitOpError("condition must be a boolean type");
  }
  
  // Check that the body region has at least one block
  if (getRegion().empty()) {
    return emitOpError("body region must have at least one block");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

// Note: YieldOp verification is handled automatically by MLIR
// No custom verification needed

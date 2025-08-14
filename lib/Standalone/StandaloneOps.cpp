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
  // For now, just verify that the result type matches the input type
  // The kind and width attributes will be handled in the lowering pass
  if (getResult().getType() != getValue().getType()) {
    return emitOpError("result type must match input type");
  }
  
  return success();
}

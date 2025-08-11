//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.cpp.inc"

using namespace mlir;
using namespace mlir::standalone;

LogicalResult AddOp::verify() {
  if (getLhs().getType() != getRhs().getType() ||
      getRes().getType() != getLhs().getType())
    return emitOpError("lhs/rhs/result types must match");
  return success();
}

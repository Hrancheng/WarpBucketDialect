//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Standalone/StandaloneTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_FooOp : Standalone_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Standalone_AddOp : Standalone_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "add two values with the same type";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$res);
  // 结果类型 = 操作数类型（由 SameOperandsAndResultType 保证/推断）
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
  let hasVerifier = 1;
}

def Standalone_WarpReduceOp : Standalone_Op<"warp_reduce", [Pure, SameOperandsAndResultType]> {
  let summary = "Warp-level reduction operation";
  let description = [{
    The `standalone.warp_reduce` operation performs a reduction across all threads in a warp.
    Supported reduction kinds: add, mul, min, max, and, or, xor.
    The width attribute specifies the warp size (32 for NVVM, 64 for ROCDL).
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

// Temporarily commented out due to type constraints - will implement later
// def Standalone_MaskedLoadOp : Standalone_Op<"masked_load", [Pure]> {
//   let summary = "Masked memory load operation";
//   let description = [{
//     The `standalone.masked_load` operation loads a value from memory only when the mask is true.
//     This is used for if-conversion of short rows to avoid divergent memory access.
//   }];
//   let arguments = (ins AnyType:$memref, AnyType:$indices, I1:$mask, AnyType:$passthrough);
//   let results = (outs AnyType:$result);
//   let assemblyFormat = "$memref `[` $indices `]` `,` $mask `,` $passthrough attr-dict `:` type($result)";
//   let hasVerifier = 1;
// }

// def Standalone_MaskedStoreOp : Standalone_Op<"masked_store", []> {
//   let summary = "Masked memory store operation";
//   let description = [{
//     The `standalone.masked_store` operation stores a value to memory only when the mask is true.
//     This is used for if-conversion of short rows to avoid divergent memory access.
//   }];
//   let arguments = (ins AnyType:$value, AnyType:$memref, AnyType:$indices, I1:$mask);
//   let assemblyFormat = "$value `,` $memref `[` $indices `]` `,` $mask attr-dict";
//   let hasVerifier = 1;
// }

def Standalone_WarpStrideLoopOp : Standalone_Op<"warp_stride_loop", []> {
  let summary = "Warp-stride loop operation";
  let description = [{
    The `standalone.warp_stride_loop` operation implements a loop where each thread in a warp
    processes elements with stride equal to warp size. This is used for long rows (len > T).
  }];
  let arguments = (ins Index:$start, Index:$end, Index:$warp_size);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$start `,` $end `,` $warp_size attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def Standalone_UniformBranchOp : Standalone_Op<"uniform_branch", []> {
  let summary = "Uniform branching operation";
  let description = [{
    The `standalone.uniform_branch` operation implements uniform branching within a kernel.
    All threads in a warp follow the same path, avoiding divergence.
  }];
  let arguments = (ins I1:$condition);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$condition $body attr-dict";
  let hasVerifier = 1;
}

def Standalone_YieldOp : Standalone_Op<"yield", [Terminator]> {
  let summary = "Yield operation for standalone dialect";
  let description = [{
    The `standalone.yield` operation yields values from regions.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
}

#endif // STANDALONE_OPS



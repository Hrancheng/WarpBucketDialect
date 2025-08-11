#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

std::unique_ptr<mlir::Pass> createLowerStandaloneToArithPass();

// 显式注册本项目所有 pass（我们先注册一个）
void registerStandalonePasses();

} // namespace standalone
} // namespace mlir

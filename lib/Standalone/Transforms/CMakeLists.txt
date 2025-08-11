add_mlir_library(MLIRStandaloneTransforms
  LowerStandaloneToArith.cpp
  LowerArithToGPU.cpp

  LINK_LIBS PUBLIC
    MLIRStandalone          # 你的方言库
    MLIRIR
    MLIRArithDialect
    MLIRGPUDialect
    MLIRTransforms          # ← 新增
    MLIRTransformUtils      # ← 建议也加上（有些版本需要）
)

add_mlir_library(MLIRStandaloneTransforms
  LowerStandaloneToArith.cpp
  LowerArithToGPU.cpp
  UniformityAnalysis.cpp
  IfConversion.cpp
  LowerWarpReduceToGPU.cpp
  LINK_LIBS PUBLIC
    MLIRStandalone
    MLIRIR
    MLIRArithDialect
    MLIRGPUDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTransforms
    MLIRTransformUtils
)

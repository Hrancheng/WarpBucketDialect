add_mlir_dialect_library(MLIRStandalone
        StandaloneTypes.cpp
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandalonePasses.cpp
        Transforms/DualPathKernelPass.cpp
        Transforms/IfConversion.cpp
        Transforms/UniformityAnalysis.cpp
        Transforms/LowerStandaloneToArith.cpp
        Transforms/LowerArithToGPU.cpp
        Transforms/LowerWarpReduceToGPU.cpp
        Transforms/AutoWarpReduceInsertion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen
        MLIRStandalonePassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRArithDialect
        MLIRGPUDialect
        MLIRSCFDialect
        MLIRControlFlowDialect
        MLIRMemRefDialect
        MLIRTransforms
        MLIRTransformUtils
        )
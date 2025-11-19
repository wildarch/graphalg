add_library(GraphAlgIR
    GraphAlgAttr.cpp
    GraphAlgCanonicalize.cpp
    GraphAlgCast.cpp
    GraphAlgDialect.cpp
    GraphAlgInterfaces.cpp
    GraphAlgOps.cpp
    GraphAlgInferDensity.cpp
    GraphAlgTypes.cpp
    SemiringTypes.cpp
)
target_include_directories(GraphAlgIR PUBLIC ../../include)
target_include_directories(GraphAlgIR SYSTEM PUBLIC ${PROJECT_BINARY_DIR}/include)
add_dependencies(GraphAlgIR
    MLIRGraphAlgOpsIncGen
    MLIRGraphAlgPassesIncGen
    MLIRGraphAlgInterfacesIncGen
)
target_link_libraries(
    GraphAlgIR
    PRIVATE
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRSupport
)
# Suppress -Wdangling-assignment-gsl for generated code from MLIR tablegen
# The warning is triggered by template instantiations in GraphAlgOps.cpp.inc
target_compile_options(GraphAlgIR PRIVATE -Wno-dangling-assignment-gsl)
if(NOT GRAPHALG_ENABLE_RTTI)
  target_compile_options(GraphAlgIR PUBLIC -fno-rtti)
endif()

add_library(GraphAlgPasses
    GraphAlgExplicateSparsity.cpp
    GraphAlgLoopAggregate.cpp
    GraphAlgPrepareInlinePass.cpp
    GraphAlgScalarizeApply.cpp
    GraphAlgSetDimensions.cpp
    GraphAlgSplitAggregate.cpp
    GraphAlgToCore.cpp
    GraphAlgToCorePipeline.cpp
    GraphAlgVerifyDimensions.cpp
)
add_dependencies(GraphAlgPasses MLIRGraphAlgPassesIncGen)
target_link_libraries(
    GraphAlgPasses
    PRIVATE
    GraphAlgAnalysis
    GraphAlgIR
    MLIRTransforms
)

add_subdirectory(analysis)
add_subdirectory(evaluate)
add_subdirectory(parse)

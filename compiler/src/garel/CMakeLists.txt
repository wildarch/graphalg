add_library(GARelIR
    GARelAttr.cpp
    GARelDialect.cpp
    GARelOps.cpp
    GARelTupleOps.cpp
    GARelTypes.cpp
)
target_include_directories(GARelIR PUBLIC ../../include)
target_include_directories(GARelIR SYSTEM PUBLIC ${PROJECT_BINARY_DIR}/include)
add_dependencies(GARelIR
    MLIRGARelOpsIncGen
)
# Suppress -Wdangling-assignment-gsl for generated code from MLIR tablegen
# The warning is triggered by template instantiations in GARelOps.cpp.inc
target_compile_options(GARelIR PRIVATE -Wno-dangling-assignment-gsl)
if(NOT GRAPHALG_ENABLE_RTTI)
  target_compile_options(GARelIR PUBLIC -fno-rtti)
endif()
target_link_libraries(
    GARelIR
    PRIVATE
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRSupport
)

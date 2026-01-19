add_library(GARelIR
    GARelAttr.cpp
    GARelDialect.cpp
    GARelOps.cpp
    GARelTypes.cpp
)
target_include_directories(GARelIR PUBLIC ../../include)
target_include_directories(GARelIR SYSTEM PUBLIC ${PROJECT_BINARY_DIR}/include)
add_dependencies(GARelIR
    MLIRGARelOpsIncGen
)
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

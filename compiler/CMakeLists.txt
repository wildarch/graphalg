# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    GraphAlgCompiler
    VERSION 0.1
    DESCRIPTION "Compiler for the GraphAlg language"
    LANGUAGES CXX)

# =============================== BEGIN OPTIONS ================================
# NOTE: If LLVM is built with RTTI disabled (the default), we must also disable
# it. If the LLVM build enables RTTI, we are free to choose.
option(GRAPHALG_ENABLE_RTTI "Enable run time type information" OFF)

option(ENABLE_WASM "Enable wasm-only targets" OFF)
# ================================ END OPTIONS =================================

# ================================= BEGIN LLVM =================================

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)

llvm_map_components_to_libnames(llvm_libs support)

# ================================== END LLVM ==================================

# ================================= BEGIN MLIR =================================
find_package(MLIR ${LLVM_PACKAGE_VERSION} REQUIRED CONFIG
    HINTS "${LLVM_INSTALL_PREFIX}/lib/cmake"
)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# Using a placeholder to set MLIR_TABLEGEN_EXE because find_package(MLIR)
# overwrites the MLIR_TABLEGEN_EXE flag
if(DEFINED SET_MLIR_TABLEGEN_PATH)
    set(MLIR_TABLEGEN_EXE "${SET_MLIR_TABLEGEN_PATH}")
endif()

include(AddMLIR)
# ================================== END MLIR ==================================

# Directories with MLIR dialects
add_subdirectory(include/graphalg)

add_subdirectory(test)

# graphalg_lib
add_library(graphalg_lib STATIC
    src/graphalg/analysis/DenseAnalysis.cpp
    src/graphalg/GraphAlgAttr.cpp
    src/graphalg/GraphAlgAttr.cpp
    src/graphalg/GraphAlgCanonicalize.cpp
    src/graphalg/GraphAlgCast.cpp
    src/graphalg/GraphAlgDialect.cpp
    src/graphalg/GraphAlgExplicateSparsity.cpp
    src/graphalg/GraphAlgInterfaces.cpp
    src/graphalg/GraphAlgLoopAggregate.cpp
    src/graphalg/GraphAlgOps.cpp
    src/graphalg/GraphAlgPrepareInlinePass.cpp
    src/graphalg/GraphAlgScalarizeApply.cpp
    src/graphalg/GraphAlgSetDimensions.cpp
    src/graphalg/GraphAlgSplitAggregate.cpp
    src/graphalg/GraphAlgToCore.cpp
    src/graphalg/GraphAlgToCorePipeline.cpp
    src/graphalg/GraphAlgTypes.cpp
    src/graphalg/GraphAlgVerifyDimensions.cpp
    src/graphalg/SemiringTypes.cpp
    src/graphalg/evaluate/Evaluator.cpp
)
target_include_directories(graphalg_lib PUBLIC include)
target_include_directories(graphalg_lib SYSTEM PUBLIC ${PROJECT_BINARY_DIR}/include)
add_dependencies(graphalg_lib
    MLIRGraphAlgOpsIncGen
    MLIRGraphAlgPassesIncGen
    MLIRGraphAlgInterfacesIncGen
)
target_link_libraries(
    graphalg_lib
    PRIVATE
    MLIRAnalysis
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRPass
    MLIRPass
    MLIRSupport
    MLIRTransforms
)
# Suppress -Wdangling-assignment-gsl for generated code from MLIR tablegen
# The warning is triggered by template instantiations in GraphAlgOps.cpp.inc
target_compile_options(graphalg_lib PRIVATE -Wno-dangling-assignment-gsl)
if(NOT GRAPHALG_ENABLE_RTTI)
  target_compile_options(graphalg_lib PUBLIC -fno-rtti)
endif()

add_library(graphalg_parse STATIC
    src/graphalg/parse/Lexer.cpp
    src/graphalg/parse/Parser.cpp
)
target_link_libraries(graphalg_parse PRIVATE graphalg_lib)

add_executable(graphalg-translate src/graphalg-translate.cpp)
target_link_libraries(graphalg-translate PRIVATE
    ${llvm_libs}
    graphalg_lib
    graphalg_parse
    MLIRTranslateLib
)

add_executable(graphalg-opt src/graphalg-opt.cpp)
target_link_libraries(graphalg-opt PRIVATE
    ${llvm_libs}
    graphalg_lib
    MLIROptLib
)

add_executable(graphalg-lsp-server src/graphalg-lsp-server.cpp)
target_link_libraries(graphalg-lsp-server PRIVATE
    ${llvm_libs}
    graphalg_lib
    MLIRLspServerLib
)

add_executable(graphalg-exec src/graphalg-exec.cpp)
target_link_libraries(graphalg-exec PRIVATE
    ${llvm_libs}
    graphalg_lib
    MLIRParser
)

add_executable(graphalg-playground src/graphalg-playground.cpp)
target_link_libraries(graphalg-playground PRIVATE
    ${llvm_libs}
    graphalg_lib
    graphalg_parse
)

if(ENABLE_WASM)
    set(PLAYGROUND_EXPORTED_FUNCTIONS
        _ga_new
        _ga_free
        _ga_parse
        _ga_diag_count
        _ga_diag_line_start
        _ga_diag_line_end
        _ga_diag_col_start
        _ga_diag_col_end
        _ga_diag_msg
        _ga_desugar
        _ga_add_arg
        _ga_set_dims
        _ga_set_arg_bool
        _ga_set_arg_int
        _ga_set_arg_real
        _ga_evaluate
        _ga_get_res_inf
        _ga_get_res_int
        _ga_get_res_real
    )
    string (REPLACE ";" "," PLAYGROUND_EXPORTED_FUNCTIONS_STR "${PLAYGROUND_EXPORTED_FUNCTIONS}")


    target_link_options(graphalg-playground PRIVATE
        -sEXPORTED_FUNCTIONS=${PLAYGROUND_EXPORTED_FUNCTIONS_STR}
        -sEXPORTED_RUNTIME_METHODS=cwrap,UTF8ToString
        -sMODULARIZE
        -sEXPORT_ES6)
endif()

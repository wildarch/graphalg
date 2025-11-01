# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    GraphAlgCompiler
    VERSION 0.1
    DESCRIPTION "Compiler for the GraphAlg language"
    LANGUAGES CXX)

# LLVM
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

# MLIR
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
    src/graphalg/GraphAlgTypes.cpp
    src/graphalg/GraphAlgVerifyDimensions.cpp
    src/graphalg/SemiringTypes.cpp
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
    graphalg_parse
)

set(ENABLE_WASM OFF CACHE BOOL "Enable wasm-only targets" FORCE)

if(ENABLE_WASM)
    add_executable(wasm_parse src/wasm_parse.cpp)
    target_link_libraries(wasm_parse PRIVATE
        ${llvm_libs}
        graphalg_lib
        graphalg_parse
    )
    target_link_options(wasm_parse PRIVATE
        -sEXPORTED_FUNCTIONS=_ga_parse
        -sEXPORTED_RUNTIME_METHODS=cwrap
        -sMODULARIZE)
endif()

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
if(DEFINED GRAPHALG_OVERRIDE_MLIR_TABLEGEN_PATH)
    set(MLIR_TABLEGEN_EXE "${GRAPHALG_OVERRIDE_MLIR_TABLEGEN_PATH}")
endif()

include(AddMLIR)
# ================================== END MLIR ==================================

# Directories with MLIR dialects
add_subdirectory(include/graphalg)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)

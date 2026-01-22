# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    pg_graphalg
    VERSION 0.1
    DESCRIPTION "GraphAlg extension for PostgreSQL"
    LANGUAGES C CXX)

find_package(PostgreSQL COMPONENTS Server)

include(FetchContent)
FetchContent_Declare(
  GraphAlg
  SOURCE_DIR ../../compiler
)
FetchContent_MakeAvailable(GraphAlg)

# ================================= BEGIN LLVM =================================

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

llvm_map_components_to_libnames(llvm_libs support)

# ================================== END LLVM ==================================



find_library(NAMES GraphBLAS)

add_subdirectory(src)

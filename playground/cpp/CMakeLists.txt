# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    GraphAlgPlayground
    VERSION 0.1
    DESCRIPTION "GraphAlg Playground - WASM side"
    LANGUAGES CXX)

option(ENABLE_WASM "Enable flags only supported during compilation to webassembly" OFF)

include(FetchContent)
FetchContent_Declare(
  GraphAlg
  SOURCE_DIR ../../../compiler
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

add_executable(graphalg-playground graphalg-playground.cpp)
target_link_libraries(graphalg-playground PRIVATE
    ${llvm_libs}
    GraphAlgEvaluate
    GraphAlgIR
    GraphAlgParse
    GraphAlgPasses
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
        _ga_get_res_ring
        _ga_get_res_rows
        _ga_get_res_cols
        _ga_get_res_bool
        _ga_get_res_int
        _ga_get_res_real
        _ga_get_res_inf
    )
    string (REPLACE ";" "," PLAYGROUND_EXPORTED_FUNCTIONS_STR "${PLAYGROUND_EXPORTED_FUNCTIONS}")

    target_link_options(graphalg-playground PRIVATE
        -sEXPORTED_FUNCTIONS=${PLAYGROUND_EXPORTED_FUNCTIONS_STR}
        -sEXPORTED_RUNTIME_METHODS=cwrap,UTF8ToString
        -sMODULARIZE
        -sEXPORT_ES6)
endif()


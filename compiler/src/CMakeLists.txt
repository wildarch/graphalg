add_subdirectory(graphalg)

add_executable(graphalg-translate graphalg-translate.cpp)
target_link_libraries(graphalg-translate PRIVATE
    ${llvm_libs}
    GraphAlgIR
    GraphAlgParse
    MLIRTranslateLib
)

add_executable(graphalg-opt graphalg-opt.cpp)
target_link_libraries(graphalg-opt PRIVATE
    ${llvm_libs}
    GraphAlgIR
    GraphAlgPasses
    MLIROptLib
)

add_executable(graphalg-lsp-server graphalg-lsp-server.cpp)
target_link_libraries(graphalg-lsp-server PRIVATE
    ${llvm_libs}
    GraphAlgIR
    MLIRLspServerLib
)

add_executable(graphalg-exec graphalg-exec.cpp)
target_link_libraries(graphalg-exec PRIVATE
    ${llvm_libs}
    GraphAlgEvaluate
    GraphAlgIR
    MLIRParser
)

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

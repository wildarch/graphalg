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

if (GRAPHALG_ENABLE_FUZZER)
    # Add fuzzer targets.
    add_executable(fuzz-parser fuzz-parser.cpp)
    target_link_libraries(fuzz-parser PRIVATE
        ${llvm_libs}
        GraphAlgIR
        GraphAlgParse
    )

    target_compile_options(fuzz-parser PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz-parser PRIVATE -fsanitize=fuzzer)
endif ()

find_program(LIT_PATH NAMES lit)
if (LIT_PATH)
    set (LLVM_EXTERNAL_LIT ${LIT_PATH} CACHE STRING "lit path" FORCE)
endif ()

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check "Run integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    graphalg-exec
    graphalg-opt
    graphalg-translate
)

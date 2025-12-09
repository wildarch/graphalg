add_library(PgGraphAlg SHARED
    MatrixTable.cpp
    PgGraphAlg.cpp
)
target_include_directories(PgGraphAlg PUBLIC ../../include)
target_link_libraries(PgGraphAlg
    PRIVATE
    ${llvm_libs}
    GraphAlgEvaluate
    GraphAlgIR
    GraphAlgParse
    GraphAlgPasses
)

add_subdirectory(pg_graphalg)

add_library(pg_graphalg SHARED
    pg_graphalg.cpp
)
target_include_directories(pg_graphalg PUBLIC ../include)
target_link_libraries(pg_graphalg
    PRIVATE
    PgGraphAlg
    PostgreSQL::PostgreSQL
    ${llvm_libs}
    GraphAlgEvaluate
    GraphAlgIR
    GraphAlgParse
    GraphAlgPasses
)

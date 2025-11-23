add_library(PgGraphAlg SHARED
    PgGraphAlg.cpp
)
target_include_directories(PgGraphAlg PUBLIC ../../include)
target_link_libraries(PgGraphAlg graphblas)

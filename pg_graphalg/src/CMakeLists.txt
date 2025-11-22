add_library(pg_graphalg SHARED
    pg_graphalg.c
)

target_link_libraries(pg_graphalg PRIVATE PostgreSQL::PostgreSQL)

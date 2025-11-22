# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    pg_graphalg
    VERSION 0.1
    DESCRIPTION "GraphAlg extension for PostgreSQL"
    LANGUAGES C CXX)

find_package(PostgreSQL COMPONENTS Server)

add_subdirectory(src)

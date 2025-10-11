# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    GraphAlgCompiler 
    VERSION 0.1
    DESCRIPTION "Compiler for the GraphAlg language"
    LANGUAGES CXX)

add_executable(parse src/parse.cpp)
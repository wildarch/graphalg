#!/bin/bash
WORKSPACE_ROOT=compiler/
BUILD_DIR=$WORKSPACE_ROOT/build-cov
rm -rf $BUILD_DIR
cmake -S $WORKSPACE_ROOT -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++-20  \
    -DCMAKE_LINKER_TYPE=MOLD \
    -DLLVM_ROOT="/opt/llvm-debug" \
    -DLLVM_ROOT="/opt/llvm-debug" \
    -DSET_LLVM_TOOLS_BINARY_DIR="/usr/lib/llvm-20/bin" \
    -DCMAKE_CXX_FLAGS="-fprofile-instr-generate=$BUILD_DIR/profiles/%4m.profraw -fcoverage-mapping" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate=$BUILD_DIR/profiles/%4m.profraw -fcoverage-mapping"

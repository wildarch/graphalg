#!/bin/bash
WORKSPACE_ROOT=playground/cpp/
BUILD_DIR=$WORKSPACE_ROOT/build-wasm
rm -rf $BUILD_DIR
emcmake cmake -S $WORKSPACE_ROOT -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_DIR="/opt/llvm-wasm/lib/cmake/llvm" \
    -DMLIR_DIR="/opt/llvm-wasm/lib/cmake/mlir" \
    -DGRAPHALG_OVERRIDE_MLIR_TABLEGEN_PATH="/usr/bin/mlir-tblgen-20" \
    -DENABLE_WASM=ON \

#!/bin/bash
WORKSPACE_ROOT=compiler/
BUILD_DIR=$WORKSPACE_ROOT/build-wasm
rm -rf $BUILD_DIR
emcmake cmake -S $WORKSPACE_ROOT -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_WASM=ON \
    -DGRAPHALG_OVERRIDE_MLIR_TABLEGEN_PATH="/usr/bin/mlir-tblgen-20" \
    -DLLVM_DIR="/opt/llvm-wasm/lib/cmake/llvm" \
    -DMLIR_DIR="/opt/llvm-wasm/lib/cmake/mlir" \

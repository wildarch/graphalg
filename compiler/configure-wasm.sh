#!/bin/bash
WORKSPACE_ROOT=compiler/
BUILD_DIR=$WORKSPACE_ROOT/build-wasm
rm -rf $BUILD_DIR
emcmake cmake -S $WORKSPACE_ROOT -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fno-rtti" \
    -DSET_MLIR_TABLEGEN_PATH="/usr/bin/mlir-tblgen-20" \
    -DLLVM_DIR="$(pwd)/third-party/llvm-wasm-install/lib/cmake/llvm" \
    -DMLIR_DIR="$(pwd)/third-party/llvm-wasm-install/lib/cmake/mlir" \
    -DENABLE_WASM=ON \

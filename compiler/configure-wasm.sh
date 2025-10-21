#!/bin/bash
WORKSPACE_ROOT=compiler/
BUILD_DIR=$WORKSPACE_ROOT/build-wasm
rm -rf $BUILD_DIR
emcmake cmake -S $WORKSPACE_ROOT -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fno-rtti"

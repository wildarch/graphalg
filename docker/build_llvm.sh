#!/bin/bash
set -e

LLVM_VERSION=20.1.2
LLVM_PROJECT_FILE_NAME=llvm-project-${LLVM_VERSION}.src
LLVM_PROJECT_SRC_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVM_PROJECT_FILE_NAME}.tar.xz"

# Download and extract sources
wget $LLVM_PROJECT_SRC_URL -O /tmp/$LLVM_PROJECT_FILE_NAME.tar.xz
sha256sum -c - <<EOF
f0a4a240aabc9b056142d14d5478bb6d962aeac549cbd75b809f5499240a8b38  /tmp/llvm-project-20.1.2.src.tar.xz
EOF

# Extract to /tmp/llvm-src
mkdir -p /tmp/llvm-src
tar -xf /tmp/$LLVM_PROJECT_FILE_NAME.tar.xz \
   -C /tmp/llvm-src --strip-components 1
rm /tmp/$LLVM_PROJECT_FILE_NAME.tar.xz

# Configure, build and install
mkdir /tmp/build/

# MLIR Debug build
# Based on https://mlir.llvm.org/getting_started/ with some tweaks
cmake -G Ninja -S /tmp/llvm-src/llvm -B /tmp/build \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_TARGETS_TO_BUILD=Native \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-debug \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_TABLEGEN=/usr/bin/mlir-tblgen-20 \
   -DLLVM_TABLEGEN=/usr/bin/llvm-tblgen-20 \
   -DMLIR_LINALG_ODS_YAML_GEN=/usr/bin/mlir-linalg-ods-yaml-gen-20 \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_ENABLE_THREADS=OFF \
   -DLLVM_ENABLE_ZLIB=OFF \
   -DLLVM_ENABLE_ZSTD=OFF \
   -DLLVM_ENABLE_CURL=OFF \
   -DLLVM_BUILD_TOOLS=OFF \
   -DLLVM_INCLUDE_BENCHMARKS=OFF \
   -DLLVM_INCLUDE_DOCS=OFF \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DLLVM_INCLUDE_UTILS=OFF \
   -DLLVM_USE_LINKER=mold \
   -DLLVM_USE_SPLIT_DWARF=ON \

cmake --build /tmp/build
sudo cmake --install /tmp/build
sudo rm -r /tmp/build/

# MLIR WASM build
source /home/ubuntu/emsdk/emsdk_env.sh
emcmake cmake -G Ninja -S /tmp/llvm-src/llvm -B /tmp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=WebAssembly \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-wasm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_TABLEGEN=/usr/bin/mlir-tblgen-20 \
    -DLLVM_TABLEGEN=/usr/bin/llvm-tblgen-20 \
    -DMLIR_LINALG_ODS_YAML_GEN=/usr/bin/mlir-linalg-ods-yaml-gen-20 \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_ENABLE_THREADS=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_CURL=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_UTILS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_ENABLE_BACKTRACES=OFF \
    -DLLVM_ENABLE_DUMP=OFF \
    -DCMAKE_CXX_FLAGS="-DLLVM_ABI=\"\" -DLLVM_TEMPLATE_ABI=\"\" -DLLVM_EXPORT_TEMPLATE=\"\""
cmake --build /tmp/build
sudo cmake --install /tmp/build
sudo rm -r /tmp/build/

# Cleanup source so it is not included in the docker image (saves ~2GiB)
rm -r /tmp/llvm-src/

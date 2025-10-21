# WASM version of GraphAlg Compiler
The goal: a GraphAlg compiler and reference runtime that can run inside a browser.

First I install emsdk: https://emscripten.org/docs/getting_started/downloads.html#

To activate the environment:

```bash
source third-party/emsdk/emsdk_env.sh
```

Try to build LLVM from source according to instructions at https://emscripten.org/docs/compiling/Building-Projects.html#integrating-with-a-build-system.

LLVM source: https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.0/llvm-project-20.1.0.src.tar.xz

General MLIR build instructions: https://mlir.llvm.org/getting_started/

Some tips taken from https://github.com/MLIR-China/mlir-playground

```bash
emcmake cmake -G Ninja -S third-party/llvm-project-20.1.0.src/llvm -B third-party/llvm-wasm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="WebAssembly" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_TABLEGEN=/usr/bin/mlir-tblgen-20 \
    -DLLVM_TABLEGEN=/usr/bin/llvm-tblgen-20 \
    -DMLIR_LINALG_ODS_YAML_GEN=/usr/bin/mlir-linalg-ods-yaml-gen-20 \
    -DLLVM_ENABLE_BACKTRACES=OFF \
    -DLLVM_ENABLE_THREADS=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_CURL=OFF \
    -DLLVM_ENABLE_DUMP=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_UTILS=OFF \
    -DCMAKE_CXX_FLAGS="-DLLVM_ABI=\"\" -DLLVM_TEMPLATE_ABI=\"\" -DLLVM_EXPORT_TEMPLATE=\"\"" \
```

I have to add a few defines for `LLVM_ABI` and friends because they do not seem to be correctly defined.

Warnings:
- ADD_LIBRARY called with SHARED option but the target platform does not
  support dynamic linking.  Building a STATIC library instead.
- If you see build failures due to cross compilation, try setting
  * HAVE_THREAD_SAFETY_ATTRIBUTES to 0
  * HAVE_POSIX_REGEX to 0
  * HAVE_STEADY_CLOCK to 0

Build the required libraries:

```bash
cmake --build third-party/llvm-wasm/ --target MLIRIR MLIRFuncDialect MLIRAnalysis MLIRPass MLIRTranslateLib
```

Install them:
```bash
cmake --install third-party/llvm-wasm/ --component MLIRIR
cmake --install third-party/llvm-wasm/ --component MLIRFuncDialect
cmake --install third-party/llvm-wasm/ --component MLIRAnalysis
cmake --install third-party/llvm-wasm/ --component MLIRPass
cmake --install third-party/llvm-wasm/ --component MLIRTranslateLib
```

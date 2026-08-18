#!/bin/bash
set -ex

cmake --build compiler/build --target graphalg-translate graphalg-opt graphalg-exec
./compiler/build/tools/graphalg-translate --import-graphalg algos/kcore.gr > /tmp/parsed.mlir
./compiler/build/tools/graphalg-opt --graphalg-to-core-pipeline --graphalg-set-dimensions='func=KCore args=16x16,1x1' /tmp/parsed.mlir > /tmp/exec.mlir
./compiler/build/tools/graphalg-exec /tmp/exec.mlir KCore algos/kcore_toy.matrix algos/kcore_k.matrix

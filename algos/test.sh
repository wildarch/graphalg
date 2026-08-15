#!/bin/bash
set -ex

cmake --build compiler/build --target graphalg-translate graphalg-opt graphalg-exec
./compiler/build/tools/graphalg-translate --import-graphalg algos/test.gr > /tmp/parsed.mlir
./compiler/build/tools/graphalg-opt --graphalg-to-core-pipeline --graphalg-set-dimensions='func=BC args=5x5,5x5' /tmp/parsed.mlir > /tmp/exec.mlir
./compiler/build/tools/graphalg-exec /tmp/exec.mlir BC algos/bc_toy.m algos/exchange.m

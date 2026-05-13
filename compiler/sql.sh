#!/bin/bash
set -e

cmake --build compiler/build --target graphalg-translate graphalg-opt garel-translate
compiler/build/tools/graphalg-translate --import-graphalg $1 | \
compiler/build/tools/graphalg-opt \
    --graphalg-to-core-pipeline \
    --graphalg-verify-loop-bounds \
    --graphalg-explicate-sparsity \
    --graphalg-split-aggregate \
    --graphalg-loop-aggregate \
    --graphalg-to-rel | \
compiler/build/tools/garel-translate --export-sql

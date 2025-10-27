#!/bin/bash
# Updates reference outputs (*.mlir.ref files) for the goldens.
set -e

BUILD_DIR="compiler/build"
GOLDEN_DIR=$(dirname $0)

for f in $GOLDEN_DIR/*.gr; do
    out="${f%.gr}.mlir.ref"
    "$BUILD_DIR/graphalg-translate" --import-graphalg < "$f" > "$out"
    echo "Wrote $out"
done

#!/bin/bash

# Updates reference outputs (*.mlir.ref files) for the goldens.

set -e

BUILD_DIR="$1"

if [[ -z "$BUILD_DIR" ]]; then
    echo "Usage: $0 <build dir>"
    exit 1
fi

GOLDEN_DIR=$(dirname $0)
PARSE="$BUILD_DIR/parse"

for f in $GOLDEN_DIR/*.gr; do
    out="${f%.gr}.mlir.ref"
    $graphalg-translate --import-graphalg "$f" > "$out"
    echo "Wrote $out"
done

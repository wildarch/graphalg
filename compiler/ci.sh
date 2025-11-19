#!/bin/bash
./compiler/configure.sh
cmake --build ./compiler/build
cmake --build ./compiler/build --target check

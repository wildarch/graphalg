Update the CHECK comments in @compiler/test/graphalg-to-rel/add.mlir to match the output of running the command `./compiler/build/tools/graphalg-opt --graphalg-to-rel compiler/test/graphalg-to-rel/add.mlir`.
Check your work by running `./compiler/build/tools/graphalg-opt --graphalg-to-rel compiler/test/graphalg-to-rel/add.mlir | FileCheck-20`

Expect to replace ipr.* ops with either arith.* or garel.*. If you think other changes are needed, check with me first.

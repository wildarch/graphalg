// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir PickAny %t/input.m | diff - %t/output.m

//--- input.m
0 0 11
0 1 12
0 2 13
1 1 14
1 2 15

//--- input.mlir
func.func @PickAny(%arg0: !graphalg.mat<3 x 3 x i64>) -> !graphalg.mat<3 x 3 x i64> {
  %0 = graphalg.pick_any %arg0 : <3 x 3 x i64>
  return %0 : !graphalg.mat<3 x 3 x i64>
}

//--- output.m
0 0 11 : i64
1 1 14 : i64

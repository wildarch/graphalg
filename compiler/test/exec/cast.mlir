// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Cast %t/input.m | diff - %t/output.m

//--- input.m
0 0 11
1 0 13
1 1 14

//--- input.mlir
func.func @Cast(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x f64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<2 x 2 x i64> -> <2 x 2 x f64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.cast_scalar %arg1 : i64 -> f64
    graphalg.apply.return %1 : f64
  }
  return %0 : !graphalg.mat<2 x 2 x f64>
}

//--- output.m
0 0 1.100000e+01 : f64
1 0 1.300000e+01 : f64
1 1 1.400000e+01 : f64

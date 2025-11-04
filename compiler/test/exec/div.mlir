// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Div %t/lhs.m %t/rhs.m | diff - %t/output.m

//--- lhs.m
0 0 42.0
0 1 0.0
1 1 42.0

//--- rhs.m
0 0 2.0
0 1 2.0
1 0 0.0
1 1 0.0

//--- input.mlir
func.func @Div(%arg0: !graphalg.mat<2 x 2 x f64>, %arg1: !graphalg.mat<2 x 2 x f64>) -> !graphalg.mat<2 x 2 x f64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<2 x 2 x f64>, !graphalg.mat<2 x 2 x f64> -> <2 x 2 x f64> {
  ^bb0(%arg2: f64, %arg3: f64):
    %1 = arith.divf %arg2, %arg3 : f64
    graphalg.apply.return %1 : f64
  }
  return %0 : !graphalg.mat<2 x 2 x f64>
}

//--- output.m
0 0 2.100000e+01 : f64

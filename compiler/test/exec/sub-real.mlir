// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Sub %t/input.m | diff - %t/output.m

//--- input.m
0 0 41.0
1 0 42.0
1 1 43.0

//--- input.mlir
func.func @Sub(%arg0: !graphalg.mat<2 x 2 x f64>) -> !graphalg.mat<2 x 2 x f64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<2 x 2 x f64> -> <2 x 2 x f64> {
  ^bb0(%arg1: f64):
    %1 = graphalg.const 42.0 : f64
    %2 = arith.subf %arg1, %1 : f64
    graphalg.apply.return %2 : f64
  }
  return %0 : !graphalg.mat<2 x 2 x f64>
}

//--- output.m
0 0 -1.000000e+00 : f64
0 1 -4.200000e+01 : f64
1 1 1.000000e+00 : f64

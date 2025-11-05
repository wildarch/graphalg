// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Add %t/input.m | diff - %t/output.m

//--- input.m
0 0 11
1 0 13
1 1 14

//--- input.mlir
func.func @Add(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<2 x 2 x i64> -> <2 x 2 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 42 : i64
    %2 = graphalg.add %arg1, %1 : i64
    graphalg.apply.return %2 : i64
  }
  return %0 : !graphalg.mat<2 x 2 x i64>
}

//--- output.m
0 0 53 : i64
0 1 42 : i64
1 0 55 : i64
1 1 56 : i64

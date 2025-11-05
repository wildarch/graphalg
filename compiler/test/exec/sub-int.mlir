// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Sub %t/input.m | diff - %t/output.m

//--- input.m
0 0 41
1 0 42
1 1 43

//--- input.mlir
func.func @Sub(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<2 x 2 x i64> -> <2 x 2 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 42 : i64
    %2 = arith.subi %arg1, %1 : i64
    graphalg.apply.return %2 : i64
  }
  return %0 : !graphalg.mat<2 x 2 x i64>
}

//--- output.m
0 0 -1 : i64
0 1 -42 : i64
1 1 1 : i64

// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Eq %t/input.m | diff - %t/output.m

//--- input.m
0 0 41
1 0 42
1 1 43

//--- input.mlir
func.func @Eq(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<2 x 2 x i64> -> <2 x 2 x i1> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 42 : i64
    %2 = graphalg.eq %arg1, %1 : i64
    graphalg.apply.return %2 : i1
  }
  return %0 : !graphalg.mat<2 x 2 x i1>
}

//--- output.m
1 0 true

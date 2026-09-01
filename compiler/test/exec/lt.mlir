// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Lt %t/input.m | diff - %t/output-lt.m

//--- input.m
0 0 41
1 0 42
1 1 43

//--- input.mlir
func.func @Lt(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<2 x 2 x i64> -> <2 x 2 x i1> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 42 : i64
    %2 = graphalg.lt %arg1, %1 : i64
    graphalg.apply.return %2 : i1
  }
  return %0 : !graphalg.mat<2 x 2 x i1>
}

//--- output-lt.m
0 0 true
0 1 true

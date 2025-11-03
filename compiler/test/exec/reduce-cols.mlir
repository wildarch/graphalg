// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir ReduceCols %t/input.m | diff - %t/output.m

//--- input.m
0 0 3
0 1 5
1 0 7
1 1 11

//--- input.mlir
func.func @ReduceCols(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<1 x 2 x i64> {
  %0 = graphalg.reduce %arg0 : <2 x 2 x i64> -> <1 x 2 x i64>
  return %0 : !graphalg.mat<1 x 2 x i64>
}

//--- output.m
0 0 10 : i64
0 1 16 : i64

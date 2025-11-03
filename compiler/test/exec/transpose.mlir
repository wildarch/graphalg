// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Transpose %t/input.m | diff - %t/output.m

//--- input.m
0 0 1
0 1 2
1 0 3
1 1 4

//--- input.mlir
func.func @Transpose(%arg0: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x i64> {
  %0 = graphalg.transpose %arg0 : <2 x 2 x i64>
  return %0 : !graphalg.mat<2 x 2 x i64>
}

//--- output.m
0 0 1 : i64
0 1 3 : i64
1 0 2 : i64
1 1 4 : i64

// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir MatMul %t/lhs.m %t/rhs.m | diff - %t/output.m

//--- lhs.m
0 0 3
0 1 5
1 0 7
1 1 11

//--- rhs.m
0 0 13
0 1 17
1 0 19
1 1 23

//--- input.mlir
func.func @MatMul(
    %arg0: !graphalg.mat<2 x 2 x i64>,
    %arg1: !graphalg.mat<2 x 2 x i64>) -> !graphalg.mat<2 x 2 x i64> {
  %0 = graphalg.mxm %arg0, %arg1 : <2 x 2 x i64>, <2 x 2 x i64>
  return %0 : !graphalg.mat<2 x 2 x i64>
}

//--- output.m
0 0 134 : i64
0 1 166 : i64
1 0 300 : i64
1 1 372 : i64

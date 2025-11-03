// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Diag %t/input.m | diff - %t/output.m

//--- input.m
0 0 42
1 0 43

//--- input.mlir
func.func @Diag(%arg0: !graphalg.mat<2 x 1 x i64>) -> !graphalg.mat<2 x 2 x i64> {
  %0 = graphalg.diag %arg0 : !graphalg.mat<2 x 1 x i64>
  return %0 : !graphalg.mat<2 x 2 x i64>
}

//--- output.m
0 0 42 : i64
1 1 43 : i64

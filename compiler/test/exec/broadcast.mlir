// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Broadcast %t/input.m | diff - %t/output.m

//--- input.m
0 0 42

//--- input.mlir
func.func @Broadcast(%arg0: !graphalg.mat<1 x 1 x i64>)
    -> !graphalg.mat<2 x 2 x i64> {
  %0 = graphalg.broadcast %arg0 : <1 x 1 x i64> -> <2 x 2 x i64>
  return %0 : !graphalg.mat<2 x 2 x i64>
}

//--- output.m
0 0 42 : i64
0 1 42 : i64
1 0 42 : i64
1 1 42 : i64

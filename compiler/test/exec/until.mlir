// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Fib | diff - %t/output.m

//--- input.mlir
func.func @Fib() -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 1 : i64 -> <1 x 1 x i64>
  %2 = graphalg.const_mat 1000000 : i64 -> <1 x 1 x i64>
  %3:2 = graphalg.for_const range(%0, %2) : <1 x 1 x i64> init(%0, %1) : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    %4 = graphalg.apply %arg1, %arg2 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
    ^bb0(%arg3: i64, %arg4: i64):
      %5 = graphalg.add %arg3, %arg4 : i64
      graphalg.apply.return %5 : i64
    }
    graphalg.yield %arg2, %4 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64>
  } until {
  ^bb0(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    %4 = graphalg.apply %arg2 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i1> {
    ^bb0(%arg3: i64):
      %5 = graphalg.const 34 : i64
      %6 = graphalg.eq %arg3, %5 : i64
      graphalg.apply.return %6 : i1
    }
    graphalg.yield %4 : !graphalg.mat<1 x 1 x i1>
  }
  return %3#1 : !graphalg.mat<1 x 1 x i64>
}

//--- output.m
0 0 34 : i64

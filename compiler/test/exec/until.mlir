// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Fib | diff - %t/output.m

//--- input.mlir
func.func @Fib() -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 1 : i64 -> <1 x 1 x i64>
  %2:2 = graphalg.for begin=0 iters=<1000000> init(%0, %1) : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    %3 = graphalg.apply %arg1, %arg2 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
    ^bb0(%arg3: i64, %arg4: i64):
      %4 = graphalg.add %arg3, %arg4 : i64
      graphalg.apply.return %4 : i64
    }
    graphalg.yield %arg2, %3 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64>
  } until {
  ^bb0(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    %3 = graphalg.apply %arg2 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i1> {
    ^bb0(%arg3: i64):
      %4 = graphalg.const 34 : i64
      %5 = graphalg.eq %arg3, %4 : i64
      graphalg.apply.return %5 : i1
    }
    graphalg.yield %3 : !graphalg.mat<1 x 1 x i1>
  }
  return %2#1 : !graphalg.mat<1 x 1 x i64>
}

//--- output.m
0 0 34 : i64

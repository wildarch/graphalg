// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Reach %t/graph.m %t/source.m | diff - %t/output.m

//--- graph.m
0 1
1 2

//--- source.m
0 0

//--- input.mlir
func.func @Reach(%arg0: !graphalg.mat<3 x 3 x i1>, %arg1: !graphalg.mat<3 x 1 x i1>) -> !graphalg.mat<3 x 1 x i1> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 3 : i64 -> <1 x 1 x i64>
  %2 = graphalg.for_const range(%0, %1) : <1 x 1 x i64> init(%arg1) : !graphalg.mat<3 x 1 x i1> -> !graphalg.mat<3 x 1 x i1> body {
  ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<3 x 1 x i1>):
    %3 = graphalg.transpose %arg0 : <3 x 3 x i1>
    %4 = graphalg.mxm %3, %arg3 : <3 x 3 x i1>, <3 x 1 x i1>
    %5 = graphalg.apply %arg3, %4 : !graphalg.mat<3 x 1 x i1>, !graphalg.mat<3 x 1 x i1> -> <3 x 1 x i1> {
    ^bb0(%arg4: i1, %arg5: i1):
      %6 = graphalg.add %arg4, %arg5 : i1
      graphalg.apply.return %6 : i1
    }
    graphalg.yield %5 : !graphalg.mat<3 x 1 x i1>
  } until {
  }
  return %2 : !graphalg.mat<3 x 1 x i1>
}

//--- output.m
0 0
1 0
2 0

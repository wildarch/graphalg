// RUN: split-file %s %t
// RUN: graphalg-exec %t/input.mlir Reach %t/graph.m %t/source.m | diff - %t/output.m

//--- graph.m
0 1
1 2

//--- source.m
0 0

//--- input.mlir
func.func @Reach(%arg0: !graphalg.mat<3 x 3 x i1>, %arg1: !graphalg.mat<3 x 1 x i1>) -> !graphalg.mat<3 x 1 x i1> {
  %0 = graphalg.for begin=0 iters=<3> init(%arg1) : !graphalg.mat<3 x 1 x i1> -> !graphalg.mat<3 x 1 x i1> body {
  ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<3 x 1 x i1>):
    %1 = graphalg.transpose %arg0 : <3 x 3 x i1>
    %2 = graphalg.mxm %1, %arg3 : <3 x 3 x i1>, <3 x 1 x i1>
    %3 = graphalg.apply %arg3, %2 : !graphalg.mat<3 x 1 x i1>, !graphalg.mat<3 x 1 x i1> -> <3 x 1 x i1> {
    ^bb0(%arg4: i1, %arg5: i1):
      %4 = graphalg.add %arg4, %arg5 : i1
      graphalg.apply.return %4 : i1
    }
    graphalg.yield %3 : !graphalg.mat<3 x 1 x i1>
  } until {
  }
  return %0 : !graphalg.mat<3 x 1 x i1>
}

//--- output.m
0 0 true
1 0 true
2 0 true

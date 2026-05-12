// RUN: graphalg-opt --graphalg-verify-loop-bounds --split-input-file --verify-diagnostics < %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @OkDim(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.for begin=0 iters=#dim init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

func.func @OkConst(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.for begin=0 iters=<42> init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

// -----
#dim = #graphalg.dim<distinct[0]<>>

func.func @DynDim(%arg0 : !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  // expected-error@below{{'graphalg.for' op loop bound is not a constant or matrix dimension}}
  %1 = graphalg.for dyn_end=%arg0 begin=0 init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

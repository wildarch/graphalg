// RUN: graphalg-opt --graphalg-verify-dimensions --split-input-file --verify-diagnostics < %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Ok(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

// -----
#dim = #graphalg.dim<distinct[0]<>>

func.func @IllegalRange() -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  // expected-error@below{{'graphalg.for_dim' op attribute "dim" has value #graphalg.dim<distinct[0]<>> which has not been marked as legal}}
  %1 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

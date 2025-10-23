// RUN: graphalg-opt --graphalg-set-dimensions='func=ForDimConst args=42x1' < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @ForDimConst(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 1 : i64 -> <1 x 1 x i64>
  // CHECK: %[[#BEGIN:]] = graphalg.const_mat 0
  // CHECK: %[[#END:]] = graphalg.const_mat 42
  // CHECK: graphalg.for_const range(%[[#BEGIN]], %[[#END]])
  %1 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

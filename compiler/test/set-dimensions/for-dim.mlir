// RUN: graphalg-opt --graphalg-set-dimensions='func=ForDimConst args=42x1' < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @ForDimConst(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 1 : i64 -> <1 x 1 x i64>
  // CHECK: graphalg.for begin=0 iters=<42>
  %1 = graphalg.for begin=0 iters=#dim init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

func.func @ForDimConst() -> !graphalg.mat<1 x 1 x i64> {

  // CHECK: %[[#ZERO:]] = graphalg.const_mat 0
  // CHECK: %[[#END:]] = graphalg.const_mat 42
  // CHECK: %[[#FOR:]] = graphalg.for_const range(%[[#ZERO]], %[[#END]])
  // CHECK-SAME: init(%[[#ZERO]])
  // CHECK:   graphalg.yield %arg1 : !graphalg.mat<1 x 1 x i64>
  // CHECK: return %[[#FOR]]

  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.for_dim range(42) init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }
  return %1 : !graphalg.mat<1 x 1 x i64>
}

// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>
// CHECK-LABEL: @ForDim
func.func @ForDim(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#INIT:]] = garel.const 42 : i64
  %0 = graphalg.const_mat 42 : i64 -> <1 x 1 x i64>

  // CHECK: %[[#BEGIN:]] = garel.const 0 : i64
  // CHECK: %[[#ITERS:]] = garel.aggregate %arg1 : <index> group_by=[] aggregators=[<COUNT>]
  // CHECK: %[[#FOR:]] = garel.for
  // CHECK-SAME: %[[#BEGIN]], %[[#INIT]] : !garel.rel<i64>, !garel.rel<i64>
  // CHECK-SAME: iters=%[[#ITERS]] result_idx=1
  %1 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    // CHECK: %[[#INC:]] = garel.project %arg2
    // CHECK: garel.for.yield %[[#INC]], %arg2
    graphalg.yield %arg1 : !graphalg.mat<1 x 1 x i64>
  } until {
  }

  // CHECK: return %[[#FOR]]
  return %1 : !graphalg.mat<1 x 1 x i64>
}

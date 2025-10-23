// RUN: graphalg-opt --graphalg-loop-aggregate < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

!vec = !graphalg.mat<#dim x 1 x f64>

func.func @AddInitReduce(%arg0: !vec) -> !vec {
  // CHECK: %[[#INIT:]] = graphalg.deferred_reduce
  // CHECK-SAME: %arg0 : !graphalg.mat<#dim x 1 x f64>
  // CHECK-SAME: -> <#dim x 1 x f64>
  //
  // CHECK: graphalg.for_dim range(#dim) init(%[[#INIT]])
  %0 = graphalg.for_dim range(#dim) init(%arg0) : !vec -> !vec body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !vec):
    %1 = graphalg.deferred_reduce %arg2 : !vec -> !vec
    graphalg.yield %1 : !vec
  } until {
  }
  return %0 : !vec
}

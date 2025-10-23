// RUN: graphalg-opt --graphalg-split-aggregate < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

func.func @Reduce(%arg0: !graphalg.mat<#dim x #dim1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
  // CHECK: %[[#REDUCE:]] = graphalg.deferred_reduce %arg0
  // CHECK-SAME: -> <#dim x 1 x i64>
  %0 = graphalg.reduce %arg0 : !graphalg.mat<#dim x #dim1 x i64> -> !graphalg.mat<#dim x 1 x i64>

  // CHECK: return %[[#REDUCE]]
  return %0 : !graphalg.mat<#dim x 1 x i64>
}

// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

// CHECK-LABEL: @TransposeTwice
func.func @TransposeTwice(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: return %arg0
  %0 = graphalg.transpose %arg0 : <#dim x #dim x i64>
  %1 = graphalg.transpose %0 : <#dim x #dim x i64>
  return %1 : !graphalg.mat<#dim x #dim x i64>
}

// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

func.func @Test(%arg0: !graphalg.mat<#dim x 1 x i64>, %arg1: !graphalg.mat<#dim x #dim1 x i64>) -> !graphalg.mat<#dim1 x 1 x i64> {
  // CHECK: %[[#TRANS:]] = graphalg.transpose %arg1
  // CHECK: %[[#MXM:]] = graphalg.mxm %[[#TRANS]], %arg0
  %0 = graphalg.vxm %arg0, %arg1 : <#dim x 1 x i64>, <#dim x #dim1 x i64>

  // CHECK: return %[[#MXM]]
  return %0 : !graphalg.mat<#dim1 x 1 x i64>
}

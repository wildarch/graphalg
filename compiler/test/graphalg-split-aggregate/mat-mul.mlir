// RUN: graphalg-opt --graphalg-split-aggregate < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>
#dim2 = #graphalg.dim<distinct[2]<>>

func.func @MatMul(%arg0: !graphalg.mat<#dim x #dim1 x i64>, %arg1: !graphalg.mat<#dim1 x #dim2 x i64>) -> !graphalg.mat<#dim x #dim2 x i64> {
  // CHECK: %[[#MXM:]] = graphalg.mxm_join %arg0, %arg1
  // CHECK: %[[#REDUCE:]] = graphalg.deferred_reduce %[[#MXM]]
  // CHECK-SAME: -> <#dim x #dim2 x i64>
  %0 = graphalg.mxm %arg0, %arg1 : <#dim x #dim1 x i64>, <#dim1 x #dim2 x i64>

  // CHECK: return %[[#REDUCE]]
  return %0 : !graphalg.mat<#dim x #dim2 x i64>
}

func.func @InnerDimOne(%arg0: !graphalg.mat<#dim x 1 x i64>, %arg1: !graphalg.mat<1 x #dim2 x i64>) -> !graphalg.mat<#dim x #dim2 x i64> {
  // CHECK: %[[#MXM:]] = graphalg.mxm_join %arg0, %arg1
  %0 = graphalg.mxm %arg0, %arg1 : <#dim x 1 x i64>, <1 x #dim2 x i64>

  // CHECK: return %[[#MXM]]
  return %0 : !graphalg.mat<#dim x #dim2 x i64>
}

// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Triu(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#PRE:]] = graphalg.transpose %arg0
  // CHECK: %[[#TRIL:]] = graphalg.tril %[[#PRE]]
  // CHECK: %[[#POST:]] = graphalg.transpose %[[#TRIL]]
  %0 = graphalg.triu %arg0 : <#dim x #dim x i64>

  // CHECK: return %[[#POST]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

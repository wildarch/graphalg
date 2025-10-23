// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Wrap(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x f64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %arg1 : i64 -> f64
  // CHECK:   graphalg.apply.return %[[#CAST]]
  %0 = graphalg.cast %arg0 : <#dim x 1 x i64> -> <#dim x 1 x f64>

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x 1 x f64>
}

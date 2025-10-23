// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @Not(%arg0: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i1>):
    // CHECK: %[[#FALSE:]] = graphalg.const false
    // CHECK: %[[#RES:]] = graphalg.eq %[[#FALSE]], %arg1
    %1 = graphalg.not %arg1

    // CHECK: graphalg.apply.return %[[#RES]]
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @NVals(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    // CHECK: %[[#ZERO:]] = graphalg.const 0
    // CHECK: %[[#FALSE:]] = graphalg.const false
    // CHECK: %[[#EQZERO:]] = graphalg.eq %[[#ZERO]], %arg1
    // CHECK: %[[#NEZERO:]] = graphalg.eq %[[#FALSE]], %[[#EQZERO]]
    // CHECK: %[[#RES:]] = graphalg.cast_scalar %[[#NEZERO]] : i1 -> i64
    %1 = graphalg.nvals %arg1 : <1 x 1 x i64>

    // CHECK: graphalg.apply.return %[[#RES]]
    graphalg.apply_inline.return %1 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

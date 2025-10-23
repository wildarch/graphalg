// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @Mask(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  %0 = graphalg.apply_inline %arg0, %arg1 : !graphalg.mat<#dim x #dim x i64>, !graphalg.mat<#dim x #dim x i1> -> <#dim x #dim x i64> {
  ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<1 x 1 x i1>):
    // CHECK: %[[#CONST:]] = graphalg.const 42
    // CHECK: %[[#FALSE:]] = graphalg.const false
    //
    // CHECK: %[[#NOT:]] = graphalg.eq %[[#FALSE]], %arg3
    // CHECK: %[[#CAST:]] = graphalg.cast_scalar %[[#NOT]] : i1 -> i64
    // CHECK: %[[#MASKED_BASE:]] = graphalg.mul %[[#CAST]], %arg2
    //
    // CHECK: %[[#CAST:]] = graphalg.cast_scalar %arg3 : i1 -> i64
    // CHECK: %[[#MASKED_INPUT:]] = graphalg.mul %[[#CAST]], %[[#CONST]]
    //
    // CHECK: %[[#RES:]] = graphalg.add %[[#MASKED_BASE]], %[[#MASKED_INPUT]]
    %1 = graphalg.literal 42 : i64
    %2 = graphalg.mask %arg2<%arg3 : <1 x 1 x i1>> = %1 : <1 x 1 x i64> {complement = false}

    // CHECK: graphalg.apply.return %[[#RES]]
    graphalg.apply_inline.return %2 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

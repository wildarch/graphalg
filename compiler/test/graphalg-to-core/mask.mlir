// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

// CHECK-LABEL: @Mask
func.func @Mask(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<#dim x #dim x i1>, %arg2: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#MASKED_BASE:]] = graphalg.apply %arg1, %arg0
  // CHECK:   %[[#FALSE:]] = graphalg.const false
  // CHECK:   %[[#NOT_MASK:]] = graphalg.eq %arg3, %[[#FALSE]]
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %[[#NOT_MASK]] : i1 -> i64
  // CHECK:   %[[#MUL:]] = graphalg.mul %[[#CAST]], %arg4
  // CHECK:   graphalg.apply.return %[[#MUL]]
  //
  // CHECK: %[[#MASKED_INPUT:]] = graphalg.apply %arg1, %arg2
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %arg3 : i1 -> i64
  // CHECK:   %[[#MUL:]] = graphalg.mul %[[#CAST]], %arg4
  // CHECK:   graphalg.apply.return %[[#MUL]]
  //
  // CHECK: %[[#MERGE:]] = graphalg.apply %[[#MASKED_BASE]], %[[#MASKED_INPUT]]
  // CHECK:   %[[#ADD:]] = graphalg.add %arg3, %arg4
  // CHECK:   graphalg.apply.return %[[#ADD]]
  %0 = graphalg.mask %arg0<%arg1 : <#dim x #dim x i1>> = %arg2 : <#dim x #dim x i64> {complement = false}

  // CHECK: return %[[#MERGE]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: @MaskComplement
func.func @MaskComplement(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<#dim x #dim x i1>, %arg2: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#MASKED_BASE:]] = graphalg.apply %arg1, %arg0
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %arg3 : i1 -> i64
  // CHECK:   %[[#MUL:]] = graphalg.mul %[[#CAST]], %arg4
  // CHECK:   graphalg.apply.return %[[#MUL]]
  //
  // CHECK: %[[#MASKED_INPUT:]] = graphalg.apply %arg1, %arg2
  // CHECK:   %[[#FALSE:]] = graphalg.const false
  // CHECK:   %[[#NOT_MASK:]] = graphalg.eq %arg3, %[[#FALSE]]
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %[[#NOT_MASK]] : i1 -> i64
  // CHECK:   %[[#MUL:]] = graphalg.mul %[[#CAST]], %arg4
  // CHECK:   graphalg.apply.return %[[#MUL]]
  //
  // CHECK: %[[#MERGE:]] = graphalg.apply %[[#MASKED_BASE]], %[[#MASKED_INPUT]]
  // CHECK:   %[[#ADD:]] = graphalg.add %arg3, %arg4
  // CHECK:   graphalg.apply.return %[[#ADD]]
  %0 = graphalg.mask %arg0<%arg1 : <#dim x #dim x i1>> = %arg2 : <#dim x #dim x i64> {complement = true}
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

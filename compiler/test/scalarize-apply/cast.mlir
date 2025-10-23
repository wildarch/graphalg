// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @Cast(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %arg1 : i64 -> i1
  // CHECK:   graphalg.apply.return %[[#CAST]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.cast %arg1 : <1 x 1 x i64> -> <1 x 1 x i1>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

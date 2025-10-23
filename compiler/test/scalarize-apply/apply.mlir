// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @ApplyInApply(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    // CHECK: %[[#MUL:]] = graphalg.mul %arg1, %arg1
    %1 = graphalg.apply %arg1, %arg1 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
    ^bb0(%arg2: i64, %arg3: i64):
      %2 = graphalg.mul %arg2, %arg3 : i64
      graphalg.apply.return %2 : i64
    }

    // CHECK: graphalg.apply.return %[[#MUL]]
    graphalg.apply_inline.return %1 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

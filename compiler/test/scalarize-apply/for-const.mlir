// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @ForConst(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    // CHECK: %[[#ONE:]] = graphalg.const 1
    // CHECK: %[[#ADD0:]] = graphalg.add %arg1, %[[#ONE]]
    // CHECK: %[[#ADD1:]] = graphalg.add %[[#ADD0]], %[[#ONE]]
    // CHECK: %[[#ADD2:]] = graphalg.add %[[#ADD1]], %[[#ONE]]
    %1 = graphalg.literal 1 : i64
    %2 = graphalg.literal 2 : i64
    %3 = graphalg.literal 5 : i64
    %4 = graphalg.for_const range(%2, %3) : <1 x 1 x i64> init(%arg1) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<1 x 1 x i64>):
      %5 = graphalg.ewise %arg3 ADD %1 : <1 x 1 x i64>
      graphalg.yield %5 : !graphalg.mat<1 x 1 x i64>
    } until {
    }

    // CHECK: graphalg.apply.return %[[#ADD2]]
    graphalg.apply_inline.return %4 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

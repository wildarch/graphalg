// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

func.func @ForDim(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    // CHECK: %[[#ONE:]] = graphalg.const 1
    // CHECK: %[[#ADD:]] = graphalg.add %arg1, %[[#ONE]]
    %1 = graphalg.literal 1 : i64
    %2 = graphalg.for_dim range(1) init(%arg1) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<1 x 1 x i64>):
      %3 = graphalg.ewise %arg3 ADD %1 : <1 x 1 x i64>
      graphalg.yield %3 : !graphalg.mat<1 x 1 x i64>
    } until {
    }

    // CHECK: graphalg.apply.return %[[#ADD]]
    graphalg.apply_inline.return %2 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

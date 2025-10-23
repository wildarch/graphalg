// RUN: graphalg-opt --graphalg-set-dimensions='func=Apply args=42x1' < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Apply(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
  // CHECK: graphalg.apply %arg0 : !graphalg.mat<42 x 1 x i64> -> <42 x 1 x i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<#dim x 1 x i64> -> <#dim x 1 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 2 : i64
    %2 = graphalg.mul %arg1, %1 : i64
    graphalg.apply.return %2 : i64
  }
  return %0 : !graphalg.mat<#dim x 1 x i64>
}

// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>
// CHECK-LABEL: @CastDim
func.func @CastDim(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#AGG:]] = garel.aggregate %arg1 : <index> group_by=[] aggregators=[<COUNT>]
  %0 = graphalg.cast_dim #dim

  // CHECK: return %[[#AGG]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

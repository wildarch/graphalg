// RUN: graphalg-opt --graphalg-prepare-inline --inline < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

func.func @test(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.literal true
  return %0 : !graphalg.mat<1 x 1 x i1>
}

func.func @Test(%arg0: !graphalg.mat<#dim x #dim1 x i64>) -> !graphalg.mat<#dim x #dim1 x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply_inline %arg0 {{.*}} -> <#dim x #dim1 x i64> {
  // CHECK:   %[[#LIT:]] = graphalg.literal true
  // CHECK:   %[[#CAST:]] = graphalg.cast %[[#LIT]] {{.*}} -> <1 x 1 x i64>
  // CHECK:   %[[#MUL:]] = graphalg.ewise %2 MUL %arg1
  // CHECK:   graphalg.apply_inline.return %[[#MUL]]
  %0 = graphalg.select_unary @test %arg0 : <#dim x #dim1 x i64>

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim1 x i64>
}

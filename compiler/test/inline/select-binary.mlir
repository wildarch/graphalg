// RUN: graphalg-opt --graphalg-prepare-inline --inline < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

func.func @test(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.ewise %arg0 EQ %arg1 : <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i1>
}

func.func @Test(%arg0: !graphalg.mat<#dim x #dim1 x i64>, %arg1 : !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<#dim x #dim1 x i64> {
  // CHECK: %[[#BROADCAST:]] = graphalg.broadcast %arg1 {{.*}} -> <#dim x #dim1 x i64>
  //
  // CHECK: %[[#APPLY:]] = graphalg.apply_inline %arg0, %[[#BROADCAST]] {{.*}} -> <#dim x #dim1 x i64> {
  // CHECK:   %[[#RES:]] = graphalg.ewise %arg2 EQ %arg3 : <1 x 1 x i64>
  // CHECK:   %[[#CAST:]] = graphalg.cast %[[#RES]] {{.*}} -> <1 x 1 x i64>
  // CHECK:   %[[#MUL:]] = graphalg.ewise %[[#CAST]] MUL %arg2
  // CHECK:   graphalg.apply_inline.return %[[#MUL]]
  %0 = graphalg.literal 42 : i64
  %1 = graphalg.select_binary @test %arg0, %arg1 : <#dim x #dim1 x i64>, <1 x 1 x i64>

  // CHECK: return %[[#APPLY]]
  return %1 : !graphalg.mat<#dim x #dim1 x i64>
}

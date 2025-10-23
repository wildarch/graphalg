// RUN: graphalg-opt --graphalg-prepare-inline --inline < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>
func.func @add(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.ewise %arg0 ADD %arg1 : <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

func.func @Test(%arg0: !graphalg.mat<#dim x #dim1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<#dim x #dim1 x i64> {
  // CHECK: %[[#BROADCAST:]] = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <#dim x #dim1 x i64>
  //
  // CHECK: %[[#APPLY:]] = graphalg.apply_inline %arg0, %[[#BROADCAST]]
  // CHECK:   -> <#dim x #dim1 x i64> {
  // CHECK:   %[[#RES:]] = graphalg.ewise %arg2 ADD %arg3 : <1 x 1 x i64>
  // CHECK:   graphalg.apply_inline.return %[[#RES]]
  %1 = graphalg.apply_binary @add %arg0, %arg1 : (!graphalg.mat<#dim x #dim1 x i64>, !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<#dim x #dim1 x i64>

  // CHECK: return %[[#APPLY]]
  return %1 : !graphalg.mat<#dim x #dim1 x i64>

}

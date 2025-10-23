// RUN: graphalg-opt --graphalg-prepare-inline --inline < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>
func.func @id(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  return %arg0 : !graphalg.mat<1 x 1 x i1>
}

func.func @Test(%arg0: !graphalg.mat<#dim x #dim1 x i1>) -> !graphalg.mat<#dim x #dim1 x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply_inline %arg0 : {{.*}} -> <#dim x #dim1 x i1> {
  // CHECK:   graphalg.apply_inline.return %arg1 : <1 x 1 x i1>
  %0 = graphalg.apply_unary @id %arg0 : <#dim x #dim1 x i1> -> <#dim x #dim1 x i1>

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim1 x i1>
}

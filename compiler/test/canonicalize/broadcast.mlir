// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

// CHECK-LABEL: @BroadcastConst
func.func @BroadcastConst(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#CONST:]] = graphalg.const_mat 42 : i64 -> <#dim x #dim x i64>
  %0 = graphalg.const_mat 42 : i64 -> <1 x 1 x i64>
  %1 = graphalg.broadcast %0 : <1 x 1 x i64> -> <#dim x #dim x i64>

  // CHECK: return %[[#CONST]]
  return %1 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: @BroadcastNoOp
func.func @BroadcastNoOp(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  %1 = graphalg.broadcast %arg0 : <#dim x #dim x i64> -> <#dim x #dim x i64>

  // CHECK: return %arg0
  return %1 : !graphalg.mat<#dim x #dim x i64>
}

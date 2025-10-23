// RUN: graphalg-opt --graphalg-set-dimensions='func=ConstMat args=42x1' < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

// CHECK: func.func @ConstMat(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64>
func.func @ConstMat(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
  // CHECK: %0 = graphalg.const_mat 42 : i64 -> <42 x 1 x i64>
  %0 = graphalg.const_mat 42 : i64 -> <#dim x 1 x i64>
  // CHECK: return %0 : !graphalg.mat<42 x 1 x i64>
  return %0 : !graphalg.mat<#dim x 1 x i64>
}

// RUN: graphalg-opt --graphalg-set-dimensions='func=GetDim args=42x1' < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>
func.func @GetDim(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: graphalg.const_mat 42 : i64
  %0 = graphalg.cast_dim #dim
  return %0 : !graphalg.mat<1 x 1 x i64>
}

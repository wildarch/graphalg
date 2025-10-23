// RUN: graphalg-opt --graphalg-split-aggregate < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

func.func @MakeDense(%arg0: !graphalg.mat<#dim x #dim1 x i64>) -> !graphalg.mat<#dim x #dim1 x i64> {
  // CHECK: %[[#CONST:]] = graphalg.const_mat 0 : i64 -> <#dim x #dim1 x i64>
  // CHECK: %[[#REDUCE:]] = graphalg.deferred_reduce %arg0, %[[#CONST]]
  // CHECK-SAME: -> <#dim x #dim1 x i64>
  %0 = graphalg.make_dense %arg0 : <#dim x #dim1 x i64> {
    %2 = graphalg.const 1 : i64
    graphalg.make_dense.return %2 : i64
  }

  // CHECK: return %[[#REDUCE]]
  return %0 : !graphalg.mat<#dim x #dim1 x i64>
}

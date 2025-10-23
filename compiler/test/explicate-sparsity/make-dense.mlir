// RUN: graphalg-opt --graphalg-explicate-sparsity < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Test(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#DENSE:]] = graphalg.make_dense %arg0
  // CHECK:   %[[#ZERO:]] = graphalg.const 0
  // CHECK:   %[[#CONST:]] = graphalg.const 42
  // CHECK:   %[[#ONE:]] = graphalg.const 1
  // CHECK:   %[[#ADD:]] = graphalg.add %[[#ZERO]], %[[#ONE]]
  // CHECK:   %[[#MUL:]] = graphalg.mul %[[#ADD]], %[[#CONST]]
  // CHECK:   graphalg.make_dense.return %[[#MUL]]
  //
  // CHECK: graphalg.apply %[[#DENSE]]
  %0 = graphalg.apply %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 42 : i64
    %2 = graphalg.const 1 : i64
    %3 = graphalg.add %arg1, %2 : i64
    %4 = graphalg.mul %3, %1 : i64
    graphalg.apply.return %4 : i64
  }
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

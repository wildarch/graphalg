// RUN: graphalg-opt --graphalg-explicate-sparsity --canonicalize < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

// CHECK-LABEL: @Add
func.func @Add(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#ADD:]] = graphalg.ewise_add %arg0, %arg1
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<#dim x #dim x i64>, !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = graphalg.add %arg2, %arg3 : i64
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#ADD]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: @AddComplex
func.func @AddComplex(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#LHS:]] = graphalg.apply %arg0
  // CHECK:   %[[#CONST:]] = graphalg.const 42
  // CHECK:   %[[#MUL:]] = graphalg.mul %arg2, %[[#CONST]]
  // CHECK:   graphalg.apply.return %[[#MUL]]
  //
  // CHECK: %[[#RHS:]] = graphalg.apply %arg1
  // CHECK:   %[[#CONST:]] = graphalg.const 42
  // CHECK:   %[[#MUL:]] = graphalg.mul %arg2, %[[#CONST]]
  // CHECK:   graphalg.apply.return %[[#MUL]]
  //
  // CHECK: %[[#ADD:]] = graphalg.ewise_add %[[#LHS]], %[[#RHS]]
  %1 = graphalg.apply %arg0, %arg1 : !graphalg.mat<#dim x #dim x i64>, !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    %2 = graphalg.const 42 : i64
    %3 = graphalg.mul %arg2, %2 : i64
    %4 = graphalg.mul %arg3, %2 : i64
    %5 = graphalg.add %3, %4 : i64
    graphalg.apply.return %5 : i64
  }

  // CHECK: return %[[#ADD]]
  return %1 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: @Sub
func.func @Sub(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg1
  // CHECK:   %[[#ZERO:]] = graphalg.const 0
  // CHECK:   %[[#NEG:]] = arith.subi %[[#ZERO]], %arg2
  // CHECK:   graphalg.apply.return %[[#NEG]]
  //
  // CHECK: %[[#ADD:]] = graphalg.ewise_add %arg0, %[[#APPLY:]]
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<#dim x #dim x i64>, !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = arith.subi %arg2, %arg3 : i64
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#ADD]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

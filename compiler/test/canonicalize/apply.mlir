// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>
!mat_i64 = !graphalg.mat<#dim x #dim x i64>

// CHECK-LABEL: @ApplyConstantOutput
func.func @ApplyConstantOutput() -> !mat_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const_mat 42 : i64 -> <#dim x #dim x i64>
  %0 = graphalg.apply -> !mat_i64 {
  ^bb0():
    %1 = graphalg.const 42 : i64
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#CONST]]
  return %0 : !mat_i64
}

// CHECK-LABEL: @ApplyConstantInput
func.func @ApplyConstantInput(%arg0: !mat_i64) -> !mat_i64 {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#CONST:]] = graphalg.const 42
  // CHECK:   %[[#MUL:]] = graphalg.mul %[[#CONST]], %arg1
  // CHECK:   graphalg.apply.return %[[#MUL]]
  %0 = graphalg.const_mat 42 : i64 -> !mat_i64
  %1 = graphalg.apply %0, %arg0 : !mat_i64, !mat_i64 -> !mat_i64 {
  ^bb0(%arg1: i64, %arg2: i64):
    %2 = graphalg.mul %arg1, %arg2 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#APPLY]]
  return %1 : !mat_i64
}

func.func @ApplyInline(%arg0 : !mat_i64, %arg1 : !mat_i64, %arg2 : !mat_i64) -> !mat_i64 {
  // CHECK: [[#APPLY:]] = graphalg.apply %arg0, %arg2, %arg1
  // CHECK:   ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
  // CHECK:   %[[#CONST:]] = graphalg.const 1 : i64
  // CHECK:   %[[#ADD0:]] = graphalg.add %[[#CONST]], %arg5 : i64
  // CHECK:   %[[#ADD1:]] = graphalg.add %arg3, %[[#ADD0]] : i64
  // CHECK:   %[[#ADD2:]] = graphalg.add %[[#ADD1]], %arg4 : i64
  // CHECK:   graphalg.apply.return %[[#ADD2]] : i64
  %0 = graphalg.apply %arg1 : !mat_i64 -> !mat_i64 {
  ^bb0(%arg3: i64):
    %1 = graphalg.const 1 : i64
    %2 = graphalg.add %1, %arg3 : i64
    graphalg.apply.return %2 : i64
  }

  %1 = graphalg.apply %arg0, %0, %arg2 : !mat_i64, !mat_i64, !mat_i64 -> !mat_i64 {
  ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
    %2 = graphalg.add %arg3, %arg4 : i64
    %3 = graphalg.add %2, %arg5 : i64
    graphalg.apply.return %3 : i64
  }

  // CHECK: return %[[#APPLY]]
  return %1 : !mat_i64
}

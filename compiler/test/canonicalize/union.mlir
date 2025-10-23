// RUN: graphalg-opt --canonicalize < %s | FileCheck %s
!mat = !graphalg.mat<42 x 43 x i64>

// CHECK-LABEL: @SingleInput
func.func @SingleInput(%arg0: !mat) -> !mat {
  // CHECK: return %arg0
  %0 = graphalg.union %arg0 : !mat -> !mat
  return %0 : !mat
}

// CHECK-LABEL: @Nested
func.func @Nested(%arg0: !mat, %arg1: !mat, %arg2: !mat) -> !mat {
  // CHECK: %[[#UNION:]] = graphalg.union %arg2, %arg0, %arg1
  %0 = graphalg.union %arg0, %arg1 : !mat, !mat -> !mat
  %1 = graphalg.union %arg2, %0 : !mat, !mat -> !mat

  // CHECK: return %[[#UNION]]
  return %1 : !mat
}

// RUN: graphalg-opt --canonicalize < %s | FileCheck %s
!int = !graphalg.mat<1 x 1 x i64>

// CHECK-LABEL: @mxm_scalar
func.func @mxm_scalar(%arg0: !int, %arg1: !int) -> !int {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#MUL:]] = graphalg.mul %arg2, %arg3
  // CHECK:   graphalg.apply.return %[[#MUL]]
  %0 = graphalg.mxm %arg0, %arg1 : !int, !int

  // CHECK: return %[[#APPLY]]
  return %0 : !int
}

// RUN: graphalg-opt --canonicalize < %s | FileCheck %s
!int = !graphalg.mat<1 x 1 x i64>

// CHECK-LABEL: @FoldConstant
func.func @FoldConstant() -> !int {
  // CHECK: %[[#CONST:]] = graphalg.const_mat 3
  %0 = graphalg.const_mat 1 : i64 -> !int
  %1 = graphalg.const_mat 2 : i64 -> !int
  %2 = graphalg.deferred_reduce %0, %1 : !int, !int -> !int

  // CHECK: return %[[#CONST]]
  return %2 : !int
}

// CHECK-LABEL: @Nested
func.func @Nested(
    %arg0: !graphalg.mat<42 x 43 x i64>,
    %arg1 : !graphalg.mat<1 x 43 x i64>,
    %arg2 : !graphalg.mat<42 x 1 x i64>)
    -> !int {
  // CHECK: %[[#RED:]] = graphalg.deferred_reduce %arg2, %arg0, %arg1
  // CHECK-SAME: -> <1 x 1 x i64>
  %0 = graphalg.deferred_reduce
    %arg0, %arg1 : !graphalg.mat<42 x 43 x i64>, !graphalg.mat<1 x 43 x i64>
    -> !graphalg.mat<1 x 43 x i64>
  %1 = graphalg.deferred_reduce
    %arg2, %0 : !graphalg.mat<42 x 1 x i64>, !graphalg.mat<1 x 43 x i64>
    -> !int

  // CHECK: return %[[#RED]]
  return %1 : !int
}

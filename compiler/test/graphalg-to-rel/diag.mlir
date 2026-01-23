// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @DiagCol
func.func @DiagCol(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: garel.remap %arg0 : <index, i64> [0, 0, 1]
  %0 = graphalg.diag %arg0 : !graphalg.mat<42 x 1 x i64>
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @DiagRow
func.func @DiagRow(%arg0: !graphalg.mat<1 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: garel.remap %arg0 : <index, i64> [0, 0, 1]
  %0 = graphalg.diag %arg0 : !graphalg.mat<1 x 42 x i64>
  return %0 : !graphalg.mat<42 x 42 x i64>
}

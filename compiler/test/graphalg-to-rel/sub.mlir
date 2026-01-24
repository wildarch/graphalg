// RUN: ag-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @SubInt
func.func @SubInt(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
  ^bb0(%arg2 : i64, %arg3: i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#SUB:]] = arith.subi %[[#LHS]], %[[#RHS]]
    %1 = arith.subi %arg2, %arg3 : i64

    // CHECK: garel.project.return %[[#SUB]]
    graphalg.apply.return %1 : i64
  }

  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @SubReal
func.func @SubReal(%arg0: !graphalg.mat<1 x 1 x f64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x f64>, !graphalg.mat<1 x 1 x f64> -> <1 x 1 x f64> {
  ^bb0(%arg2 : f64, %arg3: f64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#SUB:]] = arith.subf %[[#LHS]], %[[#RHS]]
    %1 = arith.subf %arg2, %arg3 : f64

    // CHECK: garel.project.return %[[#SUB]]
    graphalg.apply.return %1 : f64
  }

  return %0 : !graphalg.mat<1 x 1 x f64>
}

// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @DivReal
func.func @DivReal(%arg0: !graphalg.mat<1 x 1 x f64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x f64>, !graphalg.mat<1 x 1 x f64> -> <1 x 1 x f64> {
  ^bb0(%arg2 : f64, %arg3: f64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#DIV:]] = arith.divf %2, %3 : f64
    %1 = arith.divf %arg2, %arg3 : f64
    // CHECK: garel.project.return %[[#DIV]]
    graphalg.apply.return %1 : f64
  }

  return %0 : !graphalg.mat<1 x 1 x f64>
}

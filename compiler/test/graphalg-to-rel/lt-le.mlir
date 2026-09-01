// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @LtInt
func.func @LtInt(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0
    %1 = graphalg.const 0 : i64
    // CHECK: %[[#CMP:]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i64
    %2 = graphalg.lt %arg1, %1 : i64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @LtReal
func.func @LtReal(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : f64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0.000000e+00
    %1 = graphalg.const 0.000000e+00 : f64
    // CHECK: %[[#CMP:]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : f64
    %2 = graphalg.lt %arg1, %1 : f64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @LeInt
func.func @LeInt(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0
    %1 = graphalg.const 0 : i64
    // CHECK: %[[#CMP:]] = arith.cmpi sle, %[[LHS]], %[[RHS]] : i64
    %2 = graphalg.le %arg1, %1 : i64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @LeReal
func.func @LeReal(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : f64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0.000000e+00
    %1 = graphalg.const 0.000000e+00 : f64
    // CHECK: %[[#CMP:]] = arith.cmpf ole, %[[LHS]], %[[RHS]] : f64
    %2 = graphalg.le %arg1, %1 : f64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

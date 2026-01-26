// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @EqBool
func.func @EqBool(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i1> {
  ^bb0(%arg1 : i1):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant false
    %1 = graphalg.const false
    // CHECK: %[[#CMP:]] = arith.cmpi eq, %[[LHS]], %[[RHS]] : i1
    %2 = graphalg.eq %arg1, %1 : i1
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @EqInt
func.func @EqInt(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0
    %1 = graphalg.const 0 : i64
    // CHECK: %[[#CMP:]] = arith.cmpi eq, %[[LHS]], %[[RHS]] : i64
    %2 = graphalg.eq %arg1, %1 : i64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @EqReal
func.func @EqReal(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : f64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0.000000e+00
    %1 = graphalg.const 0.000000e+00 : f64
    // CHECK: %[[#CMP:]] = arith.cmpf oeq, %[[LHS]], %[[RHS]] : f64
    %2 = graphalg.eq %arg1, %1 : f64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @EqTropInt
func.func @EqTropInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0
    %1 = graphalg.const #graphalg.trop_int<0 : i64> : !graphalg.trop_i64
    // CHECK: %[[#CMP:]] = arith.cmpi eq, %[[LHS]], %[[RHS]] : i64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_i64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @EqTropReal
func.func @EqTropReal(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_f64):
    // CHECK: %[[LHS:.+]] = garel.extract 0
    // CHECK: %[[RHS:.+]] = arith.constant 0.000000e+00
    %1 = graphalg.const #graphalg.trop_float<0.0 : f64> : !graphalg.trop_f64
    // CHECK: %[[#CMP:]] = arith.cmpf oeq, %[[LHS]], %[[RHS]] : f64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_f64
    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

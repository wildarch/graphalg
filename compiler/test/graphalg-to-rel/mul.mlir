// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @MulBool
func.func @MulBool(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i1>, !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i1> {
  ^bb0(%arg2 : i1, %arg3: i1):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#MUL:]] = arith.andi %[[#LHS]], %[[#RHS]]
    // CHECK: garel.project.return %[[#MUL]]
    %1 = graphalg.mul %arg2, %arg3 : i1
    graphalg.apply.return %1 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @MulInt
func.func @MulInt(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
  ^bb0(%arg2 : i64, %arg3: i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#MUL:]] = arith.muli %[[#LHS]], %[[#RHS]]
    // CHECK: garel.project.return %[[#MUL]]
    %1 = graphalg.mul %arg2, %arg3 : i64
    graphalg.apply.return %1 : i64
  }

  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @MulReal
func.func @MulReal(%arg0: !graphalg.mat<1 x 1 x f64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x f64>, !graphalg.mat<1 x 1 x f64> -> <1 x 1 x f64> {
  ^bb0(%arg2 : f64, %arg3: f64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#MUL:]] = arith.mulf %[[#LHS]], %[[#RHS]]
    // CHECK: garel.project.return %[[#MUL]]
    %1 = graphalg.mul %arg2, %arg3 : f64
    graphalg.apply.return %1 : f64
  }

  return %0 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: @MulTropInt
func.func @MulTropInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>, !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x !graphalg.trop_i64> {
  ^bb0(%arg2 : !graphalg.trop_i64, %arg3: !graphalg.trop_i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#MUL:]] = arith.addi %[[#LHS]], %[[#RHS]]
    // CHECK: garel.project.return %[[#MUL]]
    %1 = graphalg.mul %arg2, %arg3 : !graphalg.trop_i64
    graphalg.apply.return %1 : !graphalg.trop_i64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @MulTropReal
func.func @MulTropReal(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>, !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x !graphalg.trop_f64> {
  ^bb0(%arg2 : !graphalg.trop_f64, %arg3: !graphalg.trop_f64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#MUL:]] = arith.addf %[[#LHS]], %[[#RHS]]
    // CHECK: garel.project.return %[[#MUL]]
    %1 = graphalg.mul %arg2, %arg3 : !graphalg.trop_f64
    graphalg.apply.return %1 : !graphalg.trop_f64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: @MulTropMaxInt
func.func @MulTropMaxInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>, !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> -> <1 x 1 x !graphalg.trop_max_i64> {
  ^bb0(%arg2 : !graphalg.trop_max_i64, %arg3: !graphalg.trop_max_i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#MUL:]] = arith.addi %[[#LHS]], %[[#RHS]]
    // CHECK: garel.project.return %[[#MUL]]
    %1 = graphalg.mul %arg2, %arg3 : !graphalg.trop_max_i64
    graphalg.apply.return %1 : !graphalg.trop_max_i64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

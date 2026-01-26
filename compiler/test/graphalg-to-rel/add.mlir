// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @AddBool
func.func @AddBool(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
// CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <i1, i1> -> <i1>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i1>, !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i1> {
  ^bb0(%arg2 : i1, %arg3: i1):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.ori %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : i1

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : i1
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @AddInt
func.func @AddInt(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <i64, i64> -> <i64>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
  ^bb0(%arg2 : i64, %arg3: i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.addi %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : i64

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @AddReal
func.func @AddReal(%arg0: !graphalg.mat<1 x 1 x f64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  // CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <f64, f64> -> <f64>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x f64>, !graphalg.mat<1 x 1 x f64> -> <1 x 1 x f64> {
  ^bb0(%arg2 : f64, %arg3: f64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.addf %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : f64

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : f64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: @AddTropInt
func.func @AddTropInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  // CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <i64, i64> -> <i64>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>, !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x !graphalg.trop_i64> {
  ^bb0(%arg2 : !graphalg.trop_i64, %arg3: !graphalg.trop_i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.minsi %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : !graphalg.trop_i64

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : !graphalg.trop_i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @AddTropReal
func.func @AddTropReal(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  // CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <f64, f64> -> <f64>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>, !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x !graphalg.trop_f64> {
  ^bb0(%arg2 : !graphalg.trop_f64, %arg3: !graphalg.trop_f64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.minimumf %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : !graphalg.trop_f64

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : !graphalg.trop_f64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: @AddTropMaxInt
func.func @AddTropMaxInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  // CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <i64, i64> -> <i64>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>, !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> -> <1 x 1 x !graphalg.trop_max_i64> {
  ^bb0(%arg2 : !graphalg.trop_max_i64, %arg3: !graphalg.trop_max_i64):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.maxsi %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : !graphalg.trop_max_i64

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : !graphalg.trop_max_i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

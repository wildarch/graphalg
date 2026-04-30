// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @For
func.func @For(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#BEGIN:]] = garel.const 0 : i64
  // CHECK: %[[#ITERS:]] = garel.const 10 : i64
  // CHECK: %[[#FOR:]] = garel.for %[[#BEGIN]], %arg0 : !garel.rel<i64>, !garel.rel<i64> iters=%[[#ITERS]] result_idx=1 {
  %0 = graphalg.for begin=0 iters=<10> init(%arg0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    // CHECK: %[[#PROJ:]] = garel.project %arg1
    // CHECK:   %[[#EXT:]] = garel.extract 0
    // CHECK:   %[[#ADD:]] = arith.addi %[[#EXT]], %c1_i64
    // CHECK:   garel.project.return %[[#ADD]]
    // CHECK: garel.for.yield %[[#PROJ]], %arg2
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }

  // CHECK: return %[[#FOR]]
  return %0 : !graphalg.mat<1 x 1 x i64>

}

// CHECK-LABEL: @ForResultUnused
func.func @ForResultUnused(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  // CHECK: %[[#BEGIN:]] = garel.const 0 : i64
  // CHECK: %[[#FOR:]] = garel.for %[[#BEGIN]], %arg0, %arg1
  %0:2 = graphalg.for begin=0 iters=<10> init(%arg0, %arg1) : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x f64> -> !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x f64> body {
  ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<1 x 1 x i64>, %arg4: !graphalg.mat<1 x 1 x f64>):
    // CHECK: %[[#PROJ:]] = garel.project %arg2
    // CHECK: garel.for.yield %[[#PROJ]], %arg3, %arg4
    graphalg.yield %arg3, %arg4 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x f64>
  } until {
  }

  // CHECK: return %[[#FOR]]
  return %0#1 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: @Until
func.func @Until(%arg0: !graphalg.mat<42 x 42 x i1>) -> !graphalg.mat<42 x 42 x i1> {
  // CHECK: %[[#BEGIN:]] = garel.const 0 : i64
  // CHECK: %[[#FOR:]] = garel.for %[[#BEGIN]], %arg0
  %0 = graphalg.for begin=0 iters=<10> init(%arg0) : !graphalg.mat<42 x 42 x i1> -> !graphalg.mat<42 x 42 x i1> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<42 x 42 x i1>):
    // CHECK: %[[#PROJ:]] = garel.project %arg1
    // CHECK: garel.for.yield %[[#PROJ]], %arg2
    graphalg.yield %arg2 : !graphalg.mat<42 x 42 x i1>
  // CHECK: } until {
  } until {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<42 x 42 x i1>):
    // CHECK: %[[#AGG:]] = garel.aggregate %arg2
    %1 = graphalg.deferred_reduce %arg2 : !graphalg.mat<42 x 42 x i1> -> <1 x 1 x i1>
    // CHECK: garel.for.yield %[[#AGG]]
    graphalg.yield %1 : !graphalg.mat<1 x 1 x i1>
  }
  return %0 : !graphalg.mat<42 x 42 x i1>
}

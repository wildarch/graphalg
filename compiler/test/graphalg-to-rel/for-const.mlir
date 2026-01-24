// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @ForConst
func.func @ForConst(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 10 : i64 -> <1 x 1 x i64>

  // CHECK: %[[#FOR:]] = ipr.for %arg0 {{.*}} range=[0 : 10) result_idx=0 {
  // CHECK: ^bb0(%arg1: !ipr.tuplestream<[[#IT:]]:si64>, %arg2: !ipr.tuplestream<[[#V:]]:si64>):
  // CHECK:   ipr.for.yield %arg2
  %2 = graphalg.for_const range(%0, %1) : <1 x 1 x i64> init(%arg0) : !graphalg.mat<1 x 1 x i64> -> !graphalg.mat<1 x 1 x i64> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<1 x 1 x i64>):
    graphalg.yield %arg2 : !graphalg.mat<1 x 1 x i64>
  } until {
  }

  // CHECK: return %[[#FOR]]
  return %2 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ForResultUnused
func.func @ForResultUnused(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 10 : i64 -> <1 x 1 x i64>

  // CHECK: %[[#FOR:]] = ipr.for %arg0, %arg1 {{.*}} range=[0 : 10) result_idx=1 {
  // CHECK:   ipr.for.yield %arg3, %arg4
  %2:2 = graphalg.for_const range(%0, %1) : <1 x 1 x i64> init(%arg0, %arg1) : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x f64> -> !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x f64> body {
  ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<1 x 1 x i64>, %arg4: !graphalg.mat<1 x 1 x f64>):
    graphalg.yield %arg3, %arg4 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x f64>
  } until {
  }

  // CHECK: return %[[#FOR]]
  return %2#1 : !graphalg.mat<1 x 1 x f64>
}

func.func @Until(%arg0: !graphalg.mat<42 x 42 x i1>) -> !graphalg.mat<42 x 42 x i1> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 10 : i64 -> <1 x 1 x i64>

  %2 = graphalg.for_const range(%0, %1) : <1 x 1 x i64> init(%arg0) : !graphalg.mat<42 x 42 x i1> -> !graphalg.mat<42 x 42 x i1> body {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<42 x 42 x i1>):
    graphalg.yield %arg2 : !graphalg.mat<42 x 42 x i1>
  } until {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<42 x 42 x i1>):
    %3 = graphalg.reduce %arg2 : <42 x 42 x i1> -> <1 x 1 x i1>
    graphalg.yield %3 : !graphalg.mat<1 x 1 x i1>
  }
  return %2 : !graphalg.mat<42 x 42 x i1>
}

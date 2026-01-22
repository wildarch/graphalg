// RUN: ag-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @BroadcastMat
// CHECK: %arg1: !ipr.tuplestream<[[#V:]]:si64>
func.func @BroadcastMat(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#ROWS:]] = ipr.access_vertices <[[#R:]]:!ipr.opaque_vertex>
  // CHECK-SAME: opaque_vertex=[[#R]] properties=[]
  // CHECK: %[[#COLS:]] = ipr.access_vertices <[[#C:]]:!ipr.opaque_vertex>
  // CHECK-SAME: opaque_vertex=[[#C]] properties=[]
  // CHECK: %[[#JOIN:]] = ipr.join %[[#ROWS]], %[[#COLS]], %arg1
  // CHECK:   ipr.join.return
  //
  // CHECK: %[[#RENAME:]] = ipr.rename %[[#JOIN]] {{.*}} [[[#R]], [[#C]], [[#V]]]
  %0 = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <42 x 42 x i64>

  // CHECK: return %[[#RENAME]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @BroadcastRowVec
// CHECK: %arg1: !ipr.tuplestream<[[#V:]]:si64>
func.func @BroadcastRowVec(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  // CHECK: %[[#COLS:]] = ipr.access_vertices <[[#C:]]:!ipr.opaque_vertex>
  // CHECK-SAME: opaque_vertex=[[#C]] properties=[]
  // CHECK: %[[#JOIN:]] = ipr.join %[[#COLS]], %arg1
  // CHECK:   ipr.join.return
  //
  // CHECK: %[[#RENAME:]] = ipr.rename %[[#JOIN]] {{.*}} [[[#C]], [[#V]]]
  %0 = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <1 x 42 x i64>

  // CHECK: return %[[#RENAME]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @BroadcastColVec
// CHECK: %arg1: !ipr.tuplestream<[[#V:]]:si64>
func.func @BroadcastColVec(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#ROWS:]] = ipr.access_vertices <[[#R:]]:!ipr.opaque_vertex>
  // CHECK-SAME: opaque_vertex=[[#R]] properties=[]
  // CHECK: %[[#JOIN:]] = ipr.join %[[#ROWS]], %arg1
  // CHECK:   ipr.join.return
  //
  // CHECK: %[[#RENAME:]] = ipr.rename %[[#JOIN]] {{.*}} [[[#R]], [[#V]]]
  %0 = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <42 x 1 x i64>

  // CHECK: return %[[#RENAME]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

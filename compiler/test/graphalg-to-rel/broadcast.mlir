// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @BroadcastMat
func.func @BroadcastMat(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#RNG0:]] = garel.range 42
  // CHECK: %[[#RNG1:]] = garel.range 42
  // CHECK: %[[#JOIN:]] = garel.join %[[#RNG0]], %[[#RNG1]], %arg1
  %0 = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <42 x 42 x i64>

  // CHECK: return %[[#JOIN]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @BroadcastRowVec
func.func @BroadcastRowVec(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  // CHECK: %[[#RNG:]] = garel.range 42
  // CHECK: %[[#JOIN:]] = garel.join %[[#RNG]], %arg1
  %0 = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <1 x 42 x i64>

  // CHECK: return %[[#JOIN]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @BroadcastColVec
func.func @BroadcastColVec(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#RNG:]] = garel.range 42
  // CHECK: %[[#JOIN:]] = garel.join %[[#RNG]], %arg1
  %0 = graphalg.broadcast %arg1 : <1 x 1 x i64> -> <42 x 1 x i64>

  // CHECK: return %[[#JOIN]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

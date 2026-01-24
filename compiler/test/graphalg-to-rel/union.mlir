// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @UnionMat
func.func @UnionMat(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#UNION:]] = garel.union %arg0, %arg1
  %0 = graphalg.union %arg0, %arg1 : !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64>

  // CHECK: return %[[#UNION]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @UnionRowVec
func.func @UnionRowVec(%arg0: !graphalg.mat<1 x 42 x i64>, %arg1: !graphalg.mat<1 x 42 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  // CHECK: %[[#UNION:]] = garel.union %arg0, %arg1
  %0 = graphalg.union %arg0, %arg1 : !graphalg.mat<1 x 42 x i64>, !graphalg.mat<1 x 42 x i64> -> <1 x 42 x i64>

  // CHECK: return %[[#UNION]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @UnionColVec
func.func @UnionColVec(%arg0: !graphalg.mat<42 x 1 x i64>, %arg1: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#UNION:]] = garel.union %arg0, %arg1
  %0 = graphalg.union %arg0, %arg1 : !graphalg.mat<42 x 1 x i64>, !graphalg.mat<42 x 1 x i64> -> <42 x 1 x i64>

  // CHECK: return %[[#UNION]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// CHECK-LABEL: @UnionScalar
func.func @UnionScalar(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#UNION:]] = garel.union %arg0, %arg1
  %0 = graphalg.union %arg0, %arg1 : !graphalg.mat<1 x 1 x i64>, !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64>

  // CHECK: return %[[#UNION]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

func.func @UnionFlattenRow(%arg0 : !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<1 x 43 x i64> {
  // CHECK: %[[#REMAP:]] = garel.remap %arg0 : <index, index, i64> [1, 2]
  %0 = graphalg.union %arg0 : !graphalg.mat<42 x 43 x i64> -> <1 x 43 x i64>

  // CHECK: return %[[#REMAP]]
  return %0 : !graphalg.mat<1 x 43 x i64>
}

func.func @UnionFlattenCol(%arg0 : !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#REMAP:]] = garel.remap %arg0 : <index, index, i64> [0, 2]
  %0 = graphalg.union %arg0 : !graphalg.mat<42 x 43 x i64> -> <42 x 1 x i64>

  // CHECK: return %[[#REMAP]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

func.func @UnionFlattenAll(%arg0 : !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#REMAP:]] = garel.remap %arg0 : <index, index, i64> [2]
  %0 = graphalg.union %arg0 : !graphalg.mat<42 x 43 x i64> -> <1 x 1 x i64>

  // CHECK: return %[[#REMAP]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

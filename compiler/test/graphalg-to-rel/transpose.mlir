// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @TransposeMatrix
func.func @TransposeMatrix(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<43 x 42 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <index, index, si64> -> <index, index, si64>
  // CHECK:   %[[#COL_SLOT:]] = garel.extract 1
  // CHECK:   %[[#ROW_SLOT:]] = garel.extract 0
  // CHECK:   %[[#VAL_SLOT:]] = garel.extract 2
  // CHECK:   garel.project.return %[[#COL_SLOT]], %[[#ROW_SLOT]], %[[#VAL_SLOT]]
  %0 = graphalg.transpose %arg0 : <42 x 43 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<43 x 42 x i64>
}

// CHECK-LABEL: @TransposeColVec
func.func @TransposeColVec(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <index, si64> -> <index, si64>
  // CHECK:   %[[#ROW:]] = garel.extract 0
  // CHECK:   %[[#VAL:]] = garel.extract 1
  // CHECK:   garel.project.return %[[#ROW]], %[[#VAL]] : index, si64
  %0 = graphalg.transpose %arg0 : <42 x 1 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @TransposeRowVec
func.func @TransposeRowVec(%arg0: !graphalg.mat<1 x 43 x i64>) -> !graphalg.mat<43 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <index, si64> -> <index, si64>
  // CHECK:   %[[#COL:]] = garel.extract 0
  // CHECK:   %[[#VAL:]] = garel.extract 1
  // CHECK:   garel.project.return %[[#COL]], %[[#VAL]] : index, si64
  %0 = graphalg.transpose %arg0 : <1 x 43 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<43 x 1 x i64>
}

func.func @TransposeScalar(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // NOTE: folded away
  %0 = graphalg.transpose %arg0 : <1 x 1 x i64>

  // CHECK: return %arg0
  return %0 : !graphalg.mat<1 x 1 x i64>
}

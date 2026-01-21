// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @TransposeMatrix
func.func @TransposeMatrix(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<43 x 42 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <[[ROW:#col[0-9]*]], #col1, #col2> -> <#col3, #col4, #col5> {
  // CHECK:   %[[#COL_SLOT:]] = garel.extract #col1
  // CHECK:   %[[#ROW_SLOT:]] = garel.extract [[ROW]]
  // CHECK:   %[[#VAL_SLOT:]] = garel.extract #col2
  // CHECK:   garel.project.return %[[#COL_SLOT]], %[[#ROW_SLOT]], %[[#VAL_SLOT]]
  %0 = graphalg.transpose %arg0 : <42 x 43 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<43 x 42 x i64>
}

// CHECK-LABEL: @TransposeColVec
func.func @TransposeColVec(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  %0 = graphalg.transpose %arg0 : <42 x 1 x i64>

  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @TransposeRowVec
func.func @TransposeRowVec(%arg0: !graphalg.mat<1 x 43 x i64>) -> !graphalg.mat<43 x 1 x i64> {
  %0 = graphalg.transpose %arg0 : <1 x 43 x i64>

  return %0 : !graphalg.mat<43 x 1 x i64>
}

func.func @TransposeScalar(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.transpose %arg0 : <1 x 1 x i64>

  return %0 : !graphalg.mat<1 x 1 x i64>
}

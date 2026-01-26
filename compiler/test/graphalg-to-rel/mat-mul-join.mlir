// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// === input/output shapes

// (a,b) * (b,c)
// CHECK-LABEL: @MatMulABC
func.func @MatMulABC(%arg0: !graphalg.mat<42 x 43 x i64>, %arg1: !graphalg.mat<43 x 44 x i64>) -> !graphalg.mat<42 x 44 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, index, i64>, !garel.rel<index, index, i64> [<0[1] = 1[0]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#ROW:]] = garel.extract 0
  // CHECK:   %[[#COL:]] = garel.extract 4
  // CHECK:   %[[#LHS:]] = garel.extract 2
  // CHECK:   %[[#RHS:]] = garel.extract 5
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#ROW]], %[[#COL]], %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x i64>, <43 x 44 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 44 x i64>
}

// (1,b) * (b,c)
// CHECK-LABEL: @MatMulBC
func.func @MatMulBC(%arg0: !graphalg.mat<1 x 43 x i64>, %arg1: !graphalg.mat<43 x 44 x i64>) -> !graphalg.mat<1 x 44 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, i64>, !garel.rel<index, index, i64> [<0[0] = 1[0]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#ROW:]] = garel.extract 3
  // CHECK:   %[[#LHS:]] = garel.extract 1
  // CHECK:   %[[#RHS:]] = garel.extract 4
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#ROW]], %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <1 x 43 x i64>, <43 x 44 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 44 x i64>
}

// (a,1) * (1,c)
// CHECK-LABEL: @MatMulAC
func.func @MatMulAC(%arg0: !graphalg.mat<42 x 1 x i64>, %arg1: !graphalg.mat<1 x 44 x i64>) -> !graphalg.mat<42 x 44 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, i64>, !garel.rel<index, i64> []
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#ROW:]] = garel.extract 0
  // CHECK:   %[[#COL:]] = garel.extract 2
  // CHECK:   %[[#LHS:]] = garel.extract 1
  // CHECK:   %[[#RHS:]] = garel.extract 3
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#ROW]], %[[#COL]], %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 1 x i64>, <1 x 44 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 44 x i64>
}

// (a,b) * (b,1)
// CHECK-LABEL: @MatMulAB
func.func @MatMulAB(%arg0: !graphalg.mat<42 x 43 x i64>, %arg1: !graphalg.mat<43 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, index, i64>, !garel.rel<index, i64> [<0[1] = 1[0]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#ROW:]] = garel.extract 0
  // CHECK:   %[[#LHS:]] = garel.extract 2
  // CHECK:   %[[#RHS:]] = garel.extract 4
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#ROW]], %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x i64>, <43 x 1 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// (a,1) * (1,1)
// CHECK-LABEL: @MatMulA
func.func @MatMulA(%arg0: !graphalg.mat<42 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, i64>, !garel.rel<i64> []
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#ROW:]] = garel.extract 0
  // CHECK:   %[[#LHS:]] = garel.extract 1
  // CHECK:   %[[#RHS:]] = garel.extract 2
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#ROW]], %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 1 x i64>, <1 x 1 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// (1, b) * (b, 1)
// CHECK-LABEL: @MatMulB
func.func @MatMulB(%arg0: !graphalg.mat<1 x 43 x i64>, %arg1: !graphalg.mat<43 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, i64>, !garel.rel<index, i64> [<0[0] = 1[0]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#LHS:]] = garel.extract 1
  // CHECK:   %[[#RHS:]] = garel.extract 3
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <1 x 43 x i64>, <43 x 1 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// (1,1) * (1,c)
// CHECK-LABEL: @MatMulC
func.func @MatMulC(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 44 x i64>) -> !graphalg.mat<1 x 44 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<i64>, !garel.rel<index, i64> []
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#COL:]] = garel.extract 1
  // CHECK:   %[[#LHS:]] = garel.extract 0
  // CHECK:   %[[#RHS:]] = garel.extract 2
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#COL]], %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <1 x 1 x i64>, <1 x 44 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 44 x i64>
}

// (1,1) * (1,1)
// CHECK-LABEL: @MatMulScalar
func.func @MatMulScalar(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<i64>, !garel.rel<i64> []
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  // CHECK:   %[[#LHS:]] = garel.extract 0
  // CHECK:   %[[#RHS:]] = garel.extract 1
  // CHECK:   %[[#VAL:]] = arith.muli %[[#LHS]], %[[#RHS]]
  // CHECK:   garel.project.return %[[#VAL]]
  %0 = graphalg.mxm_join %arg0, %arg1 : <1 x 1 x i64>, <1 x 1 x i64>

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// === Semirings

// CHECK-LABEL: @MatMulBool
func.func @MatMulBool(%arg0: !graphalg.mat<42 x 43 x i1>, %arg1: !graphalg.mat<43 x 44 x i1>) -> !graphalg.mat<42 x 44 x i1> {
  // CHECK: arith.andi
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x i1>, <43 x 44 x i1>
  return %0 : !graphalg.mat<42 x 44 x i1>
}

// CHECK-LABEL: @MatMulInt
func.func @MatMulInt(%arg0: !graphalg.mat<42 x 43 x i64>, %arg1: !graphalg.mat<43 x 44 x i64>) -> !graphalg.mat<42 x 44 x i64> {
  // CHECK: arith.muli
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x i64>, <43 x 44 x i64>
  return %0 : !graphalg.mat<42 x 44 x i64>
}

// CHECK-LABEL: @MatMulReal
func.func @MatMulReal(%arg0: !graphalg.mat<42 x 43 x f64>, %arg1: !graphalg.mat<43 x 44 x f64>) -> !graphalg.mat<42 x 44 x f64> {
  // CHECK: arith.mulf
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x f64>, <43 x 44 x f64>
  return %0 : !graphalg.mat<42 x 44 x f64>
}

// CHECK-LABEL: @MatMulTropInt
func.func @MatMulTropInt(%arg0: !graphalg.mat<42 x 43 x !graphalg.trop_i64>, %arg1: !graphalg.mat<43 x 44 x !graphalg.trop_i64>) -> !graphalg.mat<42 x 44 x !graphalg.trop_i64> {
  // CHECK: arith.addi
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x !graphalg.trop_i64>, <43 x 44 x !graphalg.trop_i64>
  return %0 : !graphalg.mat<42 x 44 x !graphalg.trop_i64>
}

// CHECK-LABEL: @MatMulTropReal
func.func @MatMulTropReal(%arg0: !graphalg.mat<42 x 43 x !graphalg.trop_f64>, %arg1: !graphalg.mat<43 x 44 x !graphalg.trop_f64>) -> !graphalg.mat<42 x 44 x !graphalg.trop_f64> {
  // CHECK: arith.addf
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x !graphalg.trop_f64>, <43 x 44 x !graphalg.trop_f64>
  return %0 : !graphalg.mat<42 x 44 x !graphalg.trop_f64>
}

// CHECK-LABEL: @MatMulTropMaxInt
func.func @MatMulTropMaxInt(%arg0: !graphalg.mat<42 x 43 x !graphalg.trop_max_i64>, %arg1: !graphalg.mat<43 x 44 x !graphalg.trop_max_i64>) -> !graphalg.mat<42 x 44 x !graphalg.trop_max_i64> {
  // CHECK: arith.addi
  %0 = graphalg.mxm_join %arg0, %arg1 : <42 x 43 x !graphalg.trop_max_i64>, <43 x 44 x !graphalg.trop_max_i64>
  return %0 : !graphalg.mat<42 x 44 x !graphalg.trop_max_i64>
}

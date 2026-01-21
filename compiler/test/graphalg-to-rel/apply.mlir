// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// === Arity

// CHECK-LABEL: @ApplyUnary
func.func @ApplyUnary(%arg0: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    // CHECK: %[[#VAL:]] = garel.extract 2
    // CHECK: %[[#ADD:]] = arith.addi %c1_i64, %[[#VAL]]
    %2 = graphalg.add %1, %arg1 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: %[[#COL:]] = garel.extract 1
    // CHECK: garel.project.return %[[#ROW]], %[[#COL]], %[[#ADD]]
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @ApplyBinary
func.func @ApplyBinary(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, index, i64>, !garel.rel<index, index, i64> [<0[0] = 1[0]>, <0[1] = 1[1]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    // CHECK: %[[#VAL1:]] = garel.extract 2
    // CHECK: %[[#VAL2:]] = garel.extract 5
    // CHECK: %[[#ADD:]] = arith.addi %[[#VAL1]], %[[#VAL2]]
    %1 = graphalg.add %arg2, %arg3 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: %[[#COL:]] = garel.extract 1
    // CHECK: garel.project.return %[[#ROW]], %[[#COL]], %[[#ADD]]
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @ApplyTernary
func.func @ApplyTernary(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>, %arg2: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1, %arg2 : !garel.rel<index, index, i64>, !garel.rel<index, index, i64>, !garel.rel<index, index, i64> [<0[0] = 1[0]>, <0[0] = 2[0]>, <0[1] = 1[1]>, <0[1] = 2[1]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  %0 = graphalg.apply %arg0, %arg1, %arg2 : !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
    // CHECK: %[[#VAL1:]] = garel.extract 2
    // CHECK: %[[#VAL2:]] = garel.extract 5
    // CHECK: %[[#VAL3:]] = garel.extract 8
    // CHECK: %[[#ADD1:]] = arith.addi %[[#VAL1]], %[[#VAL2]]
    %1 = graphalg.add %arg3, %arg4 : i64
    // CHECK: %[[#ADD2:]] = arith.addi %[[#ADD1]], %[[#VAL3]]
    %2 = graphalg.add %1, %arg5 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: %[[#COL:]] = garel.extract 1
    // CHECK: garel.project.return %[[#ROW]], %[[#COL]], %[[#ADD2]]
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// === Shape

// CHECK-LABEL: @ApplyMat
func.func @ApplyMat(%arg0: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0
  %0 = graphalg.apply %arg0 : !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    // CHECK: %[[#VAL:]] = garel.extract 2
    // CHECK: %[[#ADD:]] = arith.addi %c1_i64, %[[#VAL]]
    %2 = graphalg.add %1, %arg1 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: %[[#COL:]] = garel.extract 1
    // CHECK: garel.project.return %[[#ROW]], %[[#COL]], %[[#ADD]]
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @ApplyRowVec
func.func @ApplyRowVec(%arg0: !graphalg.mat<1 x 42 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 42 x i64> -> <1 x 42 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    // CHECK: %[[#VAL:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.addi %c1_i64, %[[#VAL]]
    %2 = graphalg.add %1, %arg1 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: garel.project.return %[[#ROW]], %[[#ADD]]
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @ApplyColVec
func.func @ApplyColVec(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0
  %0 = graphalg.apply %arg0 : !graphalg.mat<42 x 1 x i64> -> <42 x 1 x i64> {
  ^bb0(%arg42: i64):
    %1 = graphalg.const 1 : i64
    // CHECK: %[[#VAL:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.addi %c1_i64, %[[#VAL]]
    %2 = graphalg.add %1, %arg42 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: garel.project.return %[[#ROW]], %[[#ADD]]
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// CHECK-LABEL: @ApplyScalar
func.func @ApplyScalar(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    // CHECK: %[[#VAL:]] = garel.extract 0
    // CHECK: %[[#ADD:]] = arith.addi %c1_i64, %[[#VAL]]
    %2 = graphalg.add %1, %arg1 : i64
    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ApplyBroadcastScalar
func.func @ApplyBroadcastScalar(%arg0 : !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 43 x i64> {
  // CHECK: %[[#ROWS:]] = garel.range 42
  // CHECK: %[[#COLS:]] = garel.range 43
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %[[#ROWS]], %[[#COLS]] : !garel.rel<i64>, !garel.rel<index>, !garel.rel<index> []
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <42 x 43 x i64> {
  ^bb0(%arg1: i64):
    // CHECK: %[[#VAL:]] = garel.extract 0
    // CHECK: %[[#ROW:]] = garel.extract 1
    // CHECK: %[[#COL:]] = garel.extract 2
    // CHECK: garel.project.return %[[#ROW]], %[[#COL]], %[[#VAL]]
    graphalg.apply.return %arg1 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 43 x i64>
}

// CHECK-LABEL: @ApplyBroadcastOne
func.func @ApplyBroadcastOne(%arg0: !graphalg.mat<42 x 1 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#JOIN:]] = garel.join %arg0, %arg1 : !garel.rel<index, i64>, !garel.rel<index, index, i64> [<0[0] = 1[0]>]
  // CHECK: %[[#PROJECT:]] = garel.project %[[#JOIN]]
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<42 x 1 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    // CHECK: %[[#VAL1:]] = garel.extract 1
    // CHECK: %[[#VAL2:]] = garel.extract 4
    // CHECK: %[[#ADD:]] = arith.addi %[[#VAL1]], %[[#VAL2]]
    %1 = graphalg.add %arg2, %arg3 : i64
    // CHECK: %[[#ROW:]] = garel.extract 0
    // CHECK: %[[#COL:]] = garel.extract 3
    // CHECK: garel.project.return %[[#ROW]], %[[#COL]], %[[#ADD]]
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT:]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

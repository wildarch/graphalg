// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// === Arity

// CHECK-LABEL: @ApplyUnary
func.func @ApplyUnary(%arg0: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    %2 = graphalg.add %1, %arg1 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @ApplyBinary
func.func @ApplyBinary(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = graphalg.add %arg2, %arg3 : i64
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @ApplyTernary
func.func @ApplyTernary(%arg0: !graphalg.mat<42 x 42 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>, %arg2: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  %0 = graphalg.apply %arg0, %arg1, %arg2 : !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
    %1 = graphalg.add %arg3, %arg4 : i64
    %2 = graphalg.add %1, %arg5 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// === Shape

// CHECK-LABEL: @ApplyMat
func.func @ApplyMat(%arg0: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    %2 = graphalg.add %1, %arg1 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

// CHECK-LABEL: @ApplyRowVec
func.func @ApplyRowVec(%arg0: !graphalg.mat<1 x 42 x i64>) -> !graphalg.mat<1 x 42 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 42 x i64> -> <1 x 42 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    %2 = graphalg.add %1, %arg1 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @ApplyColVec
func.func @ApplyColVec(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<42 x 1 x i64> -> <42 x 1 x i64> {
  ^bb0(%arg42: i64):
    %1 = graphalg.const 1 : i64
    %2 = graphalg.add %1, %arg42 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// CHECK-LABEL: @ApplyScalar
func.func @ApplyScalar(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64> {
  ^bb0(%arg1: i64):
    %1 = graphalg.const 1 : i64
    %2 = graphalg.add %1, %arg1 : i64
    graphalg.apply.return %2 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ApplyBroadcastScalar
func.func @ApplyBroadcastScalar(%arg0 : !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<42 x 43 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <42 x 43 x i64> {
  ^bb0(%arg1: i64):
    graphalg.apply.return %arg1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 43 x i64>
}

// CHECK-LABEL: @ApplyBroadcastOne
func.func @ApplyBroadcastOne(%arg0: !graphalg.mat<42 x 1 x i64>, %arg1: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<42 x 1 x i64>, !graphalg.mat<42 x 42 x i64> -> <42 x 42 x i64> {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = graphalg.add %arg2, %arg3 : i64
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

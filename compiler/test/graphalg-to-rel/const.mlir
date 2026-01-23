// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// Note: The inputs for these tests have to be fairly complex because:
// - Scalar ops must be inside of an ApplyOp
// - An ApplyOp whose body is reducible to a constant value folds into a
//   ConstantMatrixOp.

// CHECK-LABEL: @ConstBool
func.func @ConstBool(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i1> {
  ^bb0(%arg1 : i1):
    // CHECK: arith.constant false
    %1 = graphalg.const false
    %2 = graphalg.eq %arg1, %1 : i1
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @ConstInt
func.func @ConstInt(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : i64):
    // CHECK: arith.constant 42 : i64
    %1 = graphalg.const 42 : i64
    %2 = graphalg.eq %arg1, %1 : i64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @ConstReal
func.func @ConstReal(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : f64):
    // CHECK: arith.constant 4.200000e+01 : f64
    %1 = graphalg.const 42.0 : f64
    %2 = graphalg.eq %arg1, %1 : f64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: ConstTropInt
func.func @ConstTropInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: arith.constant 42 : i64
    %1 = graphalg.const #graphalg.trop_int<42 : i64> : !graphalg.trop_i64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_i64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: ConstTropReal
func.func @ConstTropReal(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_f64):
    // CHECK: arith.constant 4.200000e+01 : f64
    %1 = graphalg.const #graphalg.trop_float<42.0 : f64> : !graphalg.trop_f64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_f64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: ConstTropMaxInt
func.func @ConstTropMaxInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_max_i64):
    // CHECK: arith.constant 42 : i64
    %1 = graphalg.const #graphalg.trop_int<42 : i64> : !graphalg.trop_max_i64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_max_i64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// === Infinity values

// CHECK-LABEL: ConstTropIntInf
func.func @ConstTropIntInf(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: arith.constant 9223372036854775807 : i64
    %1 = graphalg.const #graphalg.trop_inf : !graphalg.trop_i64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_i64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: ConstTropRealInf
func.func @ConstTropRealInf(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_f64):
    // CHECK: arith.constant 0x7FF0000000000000 : f64
    %1 = graphalg.const #graphalg.trop_inf : !graphalg.trop_f64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_f64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: ConstTropMaxIntInf
func.func @ConstTropMaxIntInf(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_max_i64):
    // CHECK: arith.constant -9223372036854775808 : i64
    %1 = graphalg.const #graphalg.trop_inf : !graphalg.trop_max_i64
    %2 = graphalg.eq %arg1, %1 : !graphalg.trop_max_i64
    graphalg.apply.return %2 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

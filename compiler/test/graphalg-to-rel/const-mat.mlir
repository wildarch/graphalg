// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// === Shapes

// CHECK-LABEL: @ConstMat
func.func @ConstMat() -> !graphalg.mat<42 x 43 x i64> {
  // CHECK: %[[#CONST:]] = garel.const 1 : i64
  // CHECK: %[[#ROWS:]] = garel.range 42
  // CHECK: %[[#COLS:]] = garel.range 43
  // CHECK: %[[#JOIN:]] = garel.join %[[#ROWS]], %[[#COLS]], %[[#CONST]]
  %0 = graphalg.const_mat 1 : i64 -> <42 x 43 x i64>

  // CHECK: return %[[#JOIN]]
  return %0 : !graphalg.mat<42 x 43 x i64>
}

// CHECK-LABEL: @ConstRowVec
func.func @ConstRowVec() -> !graphalg.mat<1 x 42 x i64> {
  // CHECK: %[[#CONST:]] = garel.const 1 : i64
  // CHECK: %[[#COLS:]] = garel.range 42
  // CHECK: %[[#JOIN:]] = garel.join %[[#COLS]], %[[#CONST]]
  %0 = graphalg.const_mat 1 : i64 -> <1 x 42 x i64>

  // CHECK: return %[[#JOIN]]
  return %0 : !graphalg.mat<1 x 42 x i64>
}

// CHECK-LABEL: @ConstColVec
func.func @ConstColVec() -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: %[[#CONST:]] = garel.const 1 : i64
  // CHECK: %[[#ROWS:]] = garel.range 42
  // CHECK: %[[#JOIN:]] = garel.join %[[#ROWS]], %[[#CONST]]
  %0 = graphalg.const_mat 1 : i64 -> <42 x 1 x i64>

  // CHECK: return %[[#JOIN]]
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// CHECK-LABEL: @ConstScalar
func.func @ConstScalar() -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#CONST:]] = garel.const 1 : i64
  %0 = graphalg.const_mat 1 : i64 -> <1 x 1 x i64>

  // CHECK: return %[[#CONST]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// === Semirings

// CHECK-LABEL: @ConstBool
func.func @ConstBool() -> !graphalg.mat<1 x 1 x i1> {
  // CHECK: garel.const true
  %0 = graphalg.const_mat true -> <1 x 1 x i1>
  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @ConstInt
func.func @ConstInt() -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: garel.const 1 : i64
  %0 = graphalg.const_mat 1 : i64 -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ConstReal
func.func @ConstReal() -> !graphalg.mat<1 x 1 x f64> {
  // CHECK: garel.const 1.000000e+00 : f64
  %0 = graphalg.const_mat 1.000000e+00 : f64 -> <1 x 1 x f64>
  return %0 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: ConstTropInt
func.func @ConstTropInt() -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  // CHECK: garel.const 1 : i64
  %0 = graphalg.const_mat #graphalg.trop_int<1 : i64> : !graphalg.trop_i64 -> <1 x 1 x !graphalg.trop_i64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: ConstTropReal
func.func @ConstTropReal() -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  // CHECK: garel.const 1.000000e+00 : f64
  %0 = graphalg.const_mat #graphalg.trop_float<1.000000e+00 : f64> : !graphalg.trop_f64 -> <1 x 1 x !graphalg.trop_f64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: ConstTropMaxInt
func.func @ConstTropMaxInt() -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  // CHECK: garel.const 1 : i64
  %0 = graphalg.const_mat #graphalg.trop_int<1 : i64> : !graphalg.trop_max_i64 -> <1 x 1 x !graphalg.trop_max_i64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

// === Infinity values

// CHECK-LABEL: ConstTropIntInf
func.func @ConstTropIntInf() -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  // CHECK: garel.const 9223372036854775807 : i64
  %0 = graphalg.const_mat #graphalg.trop_inf : !graphalg.trop_i64 -> <1 x 1 x !graphalg.trop_i64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: ConstTropRealInf
func.func @ConstTropRealInf() -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  // CHECK: garel.const 0x7FF0000000000000 : f64
  %0 = graphalg.const_mat #graphalg.trop_inf : !graphalg.trop_f64 -> <1 x 1 x !graphalg.trop_f64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: ConstTropMaxIntInf
func.func @ConstTropMaxIntInf() -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  // CHECK: garel.const -9223372036854775808 : i64
  %0 = graphalg.const_mat #graphalg.trop_inf : !graphalg.trop_max_i64 -> <1 x 1 x !graphalg.trop_max_i64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

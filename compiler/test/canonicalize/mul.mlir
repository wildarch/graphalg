// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

// CHECK-LABEL: @MulIdLhs
func.func @MulIdLhs(%arg0: i64) -> i64 {
  // CHECK: return %arg0
  %0 = graphalg.const 1 : i64
  %1 = graphalg.mul %arg0, %0 : i64
  return %1 : i64
}

// CHECK-LABEL: @MulIdRhs
func.func @MulIdRhs(%arg0: i64) -> i64 {
  // CHECK: return %arg0
  %0 = graphalg.const 1 : i64
  %1 = graphalg.mul %0, %arg0 : i64
  return %1 : i64
}

// CHECK-LABEL: @MulAnnihilateLhs
func.func @MulAnnihilateLhs(%arg0: i64) -> i64 {
  // CHECK: %[[#CONST:]] = graphalg.const 0
  // CHECK: return %[[#CONST]]
  %0 = graphalg.const 0 : i64
  %1 = graphalg.mul %arg0, %0 : i64
  return %1 : i64
}

// CHECK-LABEL: @MulAnnihilateRhs
func.func @MulAnnihilateRhs(%arg0: i64) -> i64 {
  // CHECK: %[[#CONST:]] = graphalg.const 0
  // CHECK: return %[[#CONST]]
  %0 = graphalg.const 0 : i64
  %1 = graphalg.mul %0, %arg0 : i64
  return %1 : i64
}

// CHECK-LABEL: @MulConstantBool
func.func @MulConstantBool() -> i1 {
  // CHECK: %[[#CONST:]] = graphalg.const false
  %0 = graphalg.const false
  %1 = graphalg.const true
  %2 = graphalg.mul %0, %1 : i1

  // CHECK: return %[[#CONST]]
  return %2 : i1
}

// CHECK-LABEL: @MulConstantInt
func.func @MulConstantInt() -> i64 {
  // CHECK: %[[#CONST:]] = graphalg.const 6 : i64
  %0 = graphalg.const 2 : i64
  %1 = graphalg.const 3 : i64
  %2 = graphalg.mul %0, %1 : i64

  // CHECK: return %[[#CONST]]
  return %2 : i64
}

// CHECK-LABEL: @MulConstantReal
func.func @MulConstantReal() -> f64 {
  // CHECK: %[[#CONST:]] = graphalg.const 6.0
  %0 = graphalg.const 2.0 : f64
  %1 = graphalg.const 3.0 : f64
  %2 = graphalg.mul %0, %1 : f64

  // CHECK: return %[[#CONST]]
  return %2 : f64
}

// CHECK-LABEL: @MulConstantTropInt
func.func @MulConstantTropInt() -> !graphalg.trop_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<3 : i64>
  %0 = graphalg.const #graphalg.trop_int<1 : i64> : !graphalg.trop_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_i64
  %2 = graphalg.mul %0, %1 : !graphalg.trop_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_i64
}

// CHECK-LABEL: @MulConstantTropIntInf
func.func @MulConstantTropIntInf() -> !graphalg.trop_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_inf
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_i64
  %2 = graphalg.mul %0, %1 : !graphalg.trop_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_i64
}

// CHECK-LABEL: @MulConstantTropReal
func.func @MulConstantTropReal() -> !graphalg.trop_f64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_float<3.000000e+00 : f64>
  %0 = graphalg.const #graphalg.trop_float<1.0 : f64> : !graphalg.trop_f64
  %1 = graphalg.const #graphalg.trop_float<2.0 : f64> : !graphalg.trop_f64
  %2 = graphalg.mul %0, %1 : !graphalg.trop_f64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_f64
}

// CHECK-LABEL: @MulConstantTropRealInf
func.func @MulConstantTropRealInf() -> !graphalg.trop_f64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_inf
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_f64
  %1 = graphalg.const #graphalg.trop_float<2.0 : f64> : !graphalg.trop_f64
  %2 = graphalg.mul %0, %1 : !graphalg.trop_f64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_f64
}

// CHECK-LABEL: @MulConstantTropMaxInt
func.func @MulConstantTropMaxInt() -> !graphalg.trop_max_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<3 : i64>
  %0 = graphalg.const #graphalg.trop_int<1 : i64> : !graphalg.trop_max_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_max_i64
  %2 = graphalg.mul %0, %1 : !graphalg.trop_max_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_max_i64
}

// CHECK-LABEL: @MulConstantTropMaxIntInf
func.func @MulConstantTropMaxIntInf() -> !graphalg.trop_max_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_inf
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_max_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_max_i64
  %2 = graphalg.mul %0, %1 : !graphalg.trop_max_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_max_i64
}

// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

// CHECK-LABEL: @AddIdLhs
func.func @AddIdLhs(%arg0: i64) -> i64 {
  // CHECK: return %arg0
  %0 = graphalg.const 0 : i64
  %1 = graphalg.add %arg0, %0 : i64
  return %1 : i64
}

// CHECK-LABEL: @AddIdRhs
func.func @AddIdRhs(%arg0: i64) -> i64 {
  // CHECK: return %arg0
  %0 = graphalg.const 0 : i64
  %1 = graphalg.add %0, %arg0 : i64
  return %1 : i64
}

// CHECK-LABEL: @AddConstantBool
func.func @AddConstantBool() -> i1 {
  // CHECK: %[[#CONST:]] = graphalg.const true
  %0 = graphalg.const false
  %1 = graphalg.const true
  %2 = graphalg.add %0, %1 : i1

  // CHECK: return %[[#CONST]]
  return %2 : i1
}

// CHECK-LABEL: @AddConstantInt
func.func @AddConstantInt() -> i64 {
  // CHECK: %[[#CONST:]] = graphalg.const 3 : i64
  %0 = graphalg.const 1 : i64
  %1 = graphalg.const 2 : i64
  %2 = graphalg.add %0, %1 : i64

  // CHECK: return %[[#CONST]]
  return %2 : i64
}

// CHECK-LABEL: @AddConstantReal
func.func @AddConstantReal() -> f64 {
  // CHECK: %[[#CONST:]] = graphalg.const 3.0
  %0 = graphalg.const 1.0 : f64
  %1 = graphalg.const 2.0 : f64
  %2 = graphalg.add %0, %1 : f64

  // CHECK: return %[[#CONST]]
  return %2 : f64
}

// CHECK-LABEL: @AddConstantTropInt
func.func @AddConstantTropInt() -> !graphalg.trop_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<1 : i64>
  %0 = graphalg.const #graphalg.trop_int<1 : i64> : !graphalg.trop_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_i64
  %2 = graphalg.add %0, %1 : !graphalg.trop_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_i64
}

// CHECK-LABEL: @AddConstantTropIntInf
func.func @AddConstantTropIntInf() -> !graphalg.trop_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<2 : i64>
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_i64
  %2 = graphalg.add %0, %1 : !graphalg.trop_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_i64
}

// CHECK-LABEL: @AddConstantTropReal
func.func @AddConstantTropReal() -> !graphalg.trop_f64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_float<1.000000e+00 : f64>
  %0 = graphalg.const #graphalg.trop_float<1.0 : f64> : !graphalg.trop_f64
  %1 = graphalg.const #graphalg.trop_float<2.0 : f64> : !graphalg.trop_f64
  %2 = graphalg.add %0, %1 : !graphalg.trop_f64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_f64
}

// CHECK-LABEL: @AddConstantTropRealInf
func.func @AddConstantTropRealInf() -> !graphalg.trop_f64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_float<2.000000e+00 : f64>
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_f64
  %1 = graphalg.const #graphalg.trop_float<2.0 : f64> : !graphalg.trop_f64
  %2 = graphalg.add %0, %1 : !graphalg.trop_f64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_f64
}

// CHECK-LABEL: @AddConstantTropMaxInt
func.func @AddConstantTropMaxInt() -> !graphalg.trop_max_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<2 : i64>
  %0 = graphalg.const #graphalg.trop_int<1 : i64> : !graphalg.trop_max_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_max_i64
  %2 = graphalg.add %0, %1 : !graphalg.trop_max_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_max_i64
}

// CHECK-LABEL: @AddConstantTropMaxIntInf
func.func @AddConstantTropMaxIntInf() -> !graphalg.trop_max_i64 {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<2 : i64>
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_max_i64
  %1 = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_max_i64
  %2 = graphalg.add %0, %1 : !graphalg.trop_max_i64

  // CHECK: return %[[#CONST]]
  return %2 : !graphalg.trop_max_i64
}

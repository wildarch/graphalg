// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

// CHECK-LABEL: @CastSame
func.func @CastSame(%arg0: i64) -> i64 {
  // CHECK: return %arg0
  %0 = graphalg.cast_scalar %arg0 : i64 -> i64
  return %0 : i64
}

// CHECK-LABEL: @CastConstFromBool
func.func @CastConstFromBool() -> (i64, i64) {
  // CHECK: %[[#ZERO:]] = graphalg.const 0 : i64
  %0 = graphalg.const false
  %1 = graphalg.cast_scalar %0 : i1 -> i64

  // CHECK: %[[#ONE:]] = graphalg.const 1 : i64
  %2 = graphalg.const true
  %3 = graphalg.cast_scalar %2 : i1 -> i64

  // CHECK: return %[[#ZERO]], %[[#ONE]]
  return %1, %3 : i64, i64
}

// CHECK-LABEL: @CastConstToBool
func.func @CastConstToBool() -> (i1, i1, i1) {
  // CHECK: %[[#FALSE:]] = graphalg.const false
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.const 0 : i64
  %1 = graphalg.cast_scalar %0 : i64 -> i1

  %2 = graphalg.const 1 : i64
  %3 = graphalg.cast_scalar %2 : i64 -> i1

  %4 = graphalg.const 42 : i64
  %5 = graphalg.cast_scalar %4 : i64 -> i1

  // CHECK: return %[[#FALSE]], %[[#TRUE]], %[[#TRUE]]
  return %1, %3, %5 : i1, i1, i1
}

// CHECK-LABEL: @CastConstIntToReal
func.func @CastConstIntToReal() -> f64 {
  // CHECK: %[[#CONST:]] = graphalg.const 4.200000e+01 : f64
  %0 = graphalg.const 42 : i64
  %1 = graphalg.cast_scalar %0 : i64 -> f64

  // CHECK: return %[[#CONST]]
  return %1 : f64
}

// CHECK-LABEL: @CastConstTropIntToTropReal
func.func @CastConstTropIntToTropReal() -> (!graphalg.trop_f64, !graphalg.trop_f64) {
  // CHECK: %[[#INF:]] = graphalg.const #graphalg.trop_inf : !graphalg.trop_f64
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_i64
  %1 = graphalg.cast_scalar %0 : !graphalg.trop_i64 -> !graphalg.trop_f64

  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_float<4.200000e+01 : f64> : !graphalg.trop_f64
  %2 = graphalg.const #graphalg.trop_int<42 : i64> : !graphalg.trop_i64
  %3 = graphalg.cast_scalar %2 : !graphalg.trop_i64 -> !graphalg.trop_f64

  // CHECK: return %[[#INF]], %[[#CONST]]
  return %1, %3 : !graphalg.trop_f64, !graphalg.trop_f64
}

// CHECK-LABEL: @CastConstRealToInt
func.func @CastConstRealToInt() -> i64 {
  // CHECK: %[[#CONST:]] = graphalg.const 2 : i64
  %0 = graphalg.const 2.99 : f64
  %1 = graphalg.cast_scalar %0 : f64 -> i64

  // CHECK: return %[[#CONST]]
  return %1 : i64
}

// CHECK-LABEL: @CastConstTropRealToTropInt
func.func @CastConstTropRealToTropInt() -> (!graphalg.trop_i64, !graphalg.trop_i64) {
  // CHECK: %[[#INF:]] = graphalg.const #graphalg.trop_inf : !graphalg.trop_i64
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_f64
  %1 = graphalg.cast_scalar %0 : !graphalg.trop_f64 -> !graphalg.trop_i64

  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<2 : i64> : !graphalg.trop_i64
  %2 = graphalg.const #graphalg.trop_float<2.99 : f64> : !graphalg.trop_f64
  %3 = graphalg.cast_scalar %2 : !graphalg.trop_f64 -> !graphalg.trop_i64

  // CHECK: return %[[#INF]], %[[#CONST]]
  return %1, %3 : !graphalg.trop_i64, !graphalg.trop_i64

}

// CHECK-LABEL: @CastConstIntToTropMaxInt
func.func @CastConstIntToTropMaxInt() -> (!graphalg.trop_max_i64, !graphalg.trop_max_i64) {
  // CHECK: %[[#CONST:]] = graphalg.const #graphalg.trop_int<42 : i64> : !graphalg.trop_max_i64
  %0 = graphalg.const 42 : i64
  %1 = graphalg.cast_scalar %0 : i64 -> !graphalg.trop_max_i64

  // CHECK: %[[#INF:]] = graphalg.const #graphalg.trop_inf : !graphalg.trop_max_i64
  %2 = graphalg.const 0 : i64
  %3 = graphalg.cast_scalar %2 : i64 -> !graphalg.trop_max_i64

  // CHECK: return %[[#CONST]], %[[#INF]] : !graphalg.trop_max_i64, !graphalg.trop_max_i64
  return %1, %3 : !graphalg.trop_max_i64, !graphalg.trop_max_i64
}

// CHECK-LABEL: @CastConstTropMaxIntToInt
func.func @CastConstTropMaxIntToInt() -> (i64, i64) {
  // CHECK: %[[#INF:]] = graphalg.const 0 : i64
  %0 = graphalg.const #graphalg.trop_inf : !graphalg.trop_max_i64
  %1 = graphalg.cast_scalar %0 : !graphalg.trop_max_i64 -> i64

  // CHECK: %[[#CONST:]] = graphalg.const 42 : i64
  %2 = graphalg.const #graphalg.trop_int<42 : i64> : !graphalg.trop_max_i64
  %3 = graphalg.cast_scalar %2 : !graphalg.trop_max_i64 -> i64

  // CHECK: return %[[#INF]], %[[#CONST]] : i64, i64
  return %1, %3 : i64, i64
}

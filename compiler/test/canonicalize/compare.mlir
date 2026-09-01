// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

// CHECK-LABEL: @LtSelf
func.func @LtSelf(%arg0: i64) -> i1 {
  // CHECK: %[[#FALSE:]] = graphalg.const false
  %0 = graphalg.lt %arg0, %arg0 : i64

  // CHECK: return %[[#FALSE]]
  return %0 : i1
}

// CHECK-LABEL: @LeSelf
func.func @LeSelf(%arg0: i64) -> i1 {
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.le %arg0, %arg0 : i64

  // CHECK: return %[[#TRUE]]
  return %0 : i1
}

// CHECK-LABEL: @LtConstTrue
func.func @LtConstTrue() -> i1 {
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.const 1 : i64
  %1 = graphalg.const 2 : i64
  %2 = graphalg.lt %0, %1 : i64

  // CHECK: return %[[#TRUE]]
  return %2 : i1
}

// CHECK-LABEL: @LtConstFalse
func.func @LtConstFalse() -> i1 {
  // CHECK: %[[#FALSE:]] = graphalg.const false
  %0 = graphalg.const 2 : i64
  %1 = graphalg.const 1 : i64
  %2 = graphalg.lt %0, %1 : i64

  // CHECK: return %[[#FALSE]]
  return %2 : i1
}

// CHECK-LABEL: @LeConstTrue
func.func @LeConstTrue() -> i1 {
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.const 2 : i64
  %1 = graphalg.const 2 : i64
  %2 = graphalg.le %0, %1 : i64

  // CHECK: return %[[#TRUE]]
  return %2 : i1
}

// CHECK-LABEL: @LeConstFalse
func.func @LeConstFalse() -> i1 {
  // CHECK: %[[#FALSE:]] = graphalg.const false
  %0 = graphalg.const 3 : i64
  %1 = graphalg.const 2 : i64
  %2 = graphalg.le %0, %1 : i64

  // CHECK: return %[[#FALSE]]
  return %2 : i1
}

// CHECK-LABEL: @LtFloat
func.func @LtFloat() -> i1 {
  // CHECK: %[[#FALSE:]] = graphalg.const false
  %0 = graphalg.const 1.5 : f64
  %1 = graphalg.const 1.0 : f64
  %2 = graphalg.lt %0, %1 : f64

  // CHECK: return %[[#FALSE]]
  return %2 : i1
}

// CHECK-LABEL: @LeFloat
func.func @LeFloat() -> i1 {
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.const 1.0 : f64
  %1 = graphalg.const 1.0 : f64
  %2 = graphalg.le %0, %1 : f64

  // CHECK: return %[[#TRUE]]
  return %2 : i1
}

// CHECK-LABEL: @LtNotFolded
func.func @LtNotFolded(%arg0: i64, %arg1: i64) -> i1 {
  // CHECK: %[[#LT:]] = graphalg.lt %arg0, %arg1 : i64
  %0 = graphalg.lt %arg0, %arg1 : i64

  // CHECK: return %[[#LT]]
  return %0 : i1
}

// CHECK-LABEL: @LeNotFolded
func.func @LeNotFolded(%arg0: i64, %arg1: i64) -> i1 {
  // CHECK: %[[#LE:]] = graphalg.le %arg0, %arg1 : i64
  %0 = graphalg.le %arg0, %arg1 : i64

  // CHECK: return %[[#LE]]
  return %0 : i1
}

// CHECK-LABEL: @NotLt
func.func @NotLt(%arg0: i64, %arg1: i64) -> i1 {
  // CHECK: %[[#LE:]] = graphalg.le %arg1, %arg0 : i64
  %0 = graphalg.lt %arg0, %arg1 : i64
  %1 = graphalg.const false
  %2 = graphalg.eq %1, %0 : i1

  // CHECK: return %[[#LE]]
  return %2 : i1
}

// CHECK-LABEL: @NotLe
func.func @NotLe(%arg0: i64, %arg1: i64) -> i1 {
  // CHECK: %[[#LT:]] = graphalg.lt %arg1, %arg0 : i64
  %0 = graphalg.le %arg0, %arg1 : i64
  %1 = graphalg.const false
  %2 = graphalg.eq %0, %1 : i1

  // CHECK: return %[[#LT]]
  return %2 : i1
}

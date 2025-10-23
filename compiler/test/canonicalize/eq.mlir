// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

// CHECK-LABEL: @EqSelf
func.func @EqSelf(%arg0: i64) -> i1 {
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.eq %arg0, %arg0 : i64

  // CHECK: return %[[#TRUE]]
  return %0 : i1
}

// CHECK-LABEL: @EqSameConst
func.func @EqSameConst() -> i1 {
  // CHECK: %[[#TRUE:]] = graphalg.const true
  %0 = graphalg.const 0 : i64
  %1 = graphalg.const 0 : i64
  %2 = graphalg.eq %0, %1 : i64

  // CHECK: return %[[#TRUE]]
  return %2 : i1
}

// CHECK-LABEL: @EqDifferentConst
func.func @EqDifferentConst() -> i1 {
  // CHECK: %[[#FALSE:]] = graphalg.const false
  %0 = graphalg.const 0 : i64
  %1 = graphalg.const 1 : i64
  %2 = graphalg.eq %0, %1 : i64

  // CHECK: return %[[#FALSE]]
  return %2 : i1
}

func.func @EqNotNot(%arg0: i1) -> i1 {
  %0 = graphalg.const false
  %1 = graphalg.eq %arg0, %0 : i1
  %2 = graphalg.eq %1, %0 : i1

  // CHECK: return %arg0
  return %2 : i1
}

// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

!int = !graphalg.mat<#dim x #dim x i64>
!real = !graphalg.mat<#dim x #dim x f64>
!bool = !graphalg.mat<#dim x #dim x i1>

// CHECK-LABEL: @add
func.func @add(%arg0: !int, %arg1: !int) -> !int {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#ADD:]] = graphalg.add %arg2, %arg3
  // CHECK:   graphalg.apply.return %[[#ADD]]
  %0 = graphalg.ewise %arg0 ADD %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !int
}

// CHECK-LABEL: @sub
func.func @sub(%arg0: !int, %arg1: !int) -> !int {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#SUB:]] = arith.subi %arg2, %arg3
  // CHECK:   graphalg.apply.return %[[#SUB]]
  %0 = graphalg.ewise %arg0 SUB %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !int
}

// CHECK-LABEL: @mul
func.func @mul(%arg0: !int, %arg1: !int) -> !int {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#MUL:]] = graphalg.mul %arg2, %arg3
  // CHECK:   graphalg.apply.return %[[#MUL]]
  %0 = graphalg.ewise %arg0 MUL %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !int
}

// CHECK-LABEL: @div
func.func @div(%arg0: !real, %arg1: !real) -> !real {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#DIV:]] = arith.divf %arg2, %arg3
  // CHECK:   graphalg.apply.return %[[#DIV]]
  %0 = graphalg.ewise %arg0 DIV %arg1 : !real

  // CHECK: return %[[#APPLY]]
  return %0 : !real
}

// CHECK-LABEL: @eq
func.func @eq(%arg0: !int, %arg1: !int) -> !bool {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#EQ:]] = graphalg.eq %arg2, %arg3
  // CHECK:   graphalg.apply.return %[[#EQ]]
  %0 = graphalg.ewise %arg0 EQ %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !bool
}

// CHECK-LABEL: @ne
func.func @ne(%arg0: !int, %arg1: !int) -> !bool {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#FALSE:]] = graphalg.const false
  // CHECK:   %[[#EQ:]] = graphalg.eq %arg2, %arg3
  // CHECK:   %[[#NE:]] = graphalg.eq %1, %2
  // CHECK:   graphalg.apply.return %[[#NE]]
  %0 = graphalg.ewise %arg0 NE %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !bool
}

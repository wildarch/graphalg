// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

!int = !graphalg.mat<#dim x #dim x i64>
!bool = !graphalg.mat<#dim x #dim x i1>

// CHECK-LABEL: @lt
func.func @lt(%arg0: !int, %arg1: !int) -> !bool {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#LT:]] = graphalg.lt %arg2, %arg3 : i64
  // CHECK:   graphalg.apply.return %[[#LT]]
  %0 = graphalg.ewise %arg0 LT %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !bool
}

// CHECK-LABEL: @gt
func.func @gt(%arg0: !int, %arg1: !int) -> !bool {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#LT:]] = graphalg.lt %arg3, %arg2 : i64
  // CHECK:   graphalg.apply.return %[[#LT]]
  %0 = graphalg.ewise %arg0 GT %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !bool
}

// CHECK-LABEL: @le
func.func @le(%arg0: !int, %arg1: !int) -> !bool {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#LE:]] = graphalg.le %arg2, %arg3 : i64
  // CHECK:   graphalg.apply.return %[[#LE]]
  %0 = graphalg.ewise %arg0 LE %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !bool
}

// CHECK-LABEL: @ge
func.func @ge(%arg0: !int, %arg1: !int) -> !bool {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0, %arg1
  // CHECK:   %[[#LE:]] = graphalg.le %arg3, %arg2 : i64
  // CHECK:   graphalg.apply.return %[[#LE]]
  %0 = graphalg.ewise %arg0 GE %arg1 : !int

  // CHECK: return %[[#APPLY]]
  return %0 : !bool
}

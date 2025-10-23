// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s

func.func @Not(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#FALSE:]] = graphalg.const false
  // CHECK:   %[[#NOT:]] = graphalg.eq %arg1, %[[#FALSE]] : i1
  // CHECK:   graphalg.apply.return %[[#NOT]]
  %0 = graphalg.not %arg0

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<1 x 1 x i1>
}

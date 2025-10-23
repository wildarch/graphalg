// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s

func.func @Neg(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#ZERO:]] = graphalg.const 0.000000e+00
  // CHECK:   %[[#SUB:]] = arith.subf %[[#ZERO]], %arg1
  // CHECK:   graphalg.apply.return %[[#SUB]]
  %0 = graphalg.neg %arg0 : <1 x 1 x f64>

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<1 x 1 x f64>
}

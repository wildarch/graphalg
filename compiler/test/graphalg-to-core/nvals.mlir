// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Nvals(%arg0: !graphalg.mat<#dim x #dim x f64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#ZERO:]] = graphalg.const 0.000000e+00 : f64
  // CHECK:   %[[#EQZERO:]] = graphalg.eq %arg1, %[[#ZERO]]
  // CHECK:   %[[#FALSE:]] = graphalg.const false
  // CHECK:   %[[#NEZERO:]] = graphalg.eq %[[#EQZERO]], %[[#FALSE]]
  // CHECK:   %[[#CAST:]] = graphalg.cast_scalar %[[#NEZERO]] {{.*}} -> i64
  // CHECK:   graphalg.apply.return %[[#CAST]]
  //
  // CHECK: %[[#REDUCE:]] = graphalg.reduce %[[#APPLY]]
  %0 = graphalg.nvals %arg0 : <#dim x #dim x f64>

  // CHECK: return %[[#REDUCE]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

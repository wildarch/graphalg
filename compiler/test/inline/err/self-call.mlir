// RUN: graphalg-opt %s --graphalg-prepare-inline --verify-diagnostics

// expected-error@below {{The program contains a cycle}}
// expected-note@below {{Part of the cycle}}
func.func @SelfCall(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.literal 1 : i64
  %1 = graphalg.apply_unary @SelfCall %arg0 : <1 x 1 x i64> -> <1 x 1 x i64>
  %2 = graphalg.ewise %0 ADD %1 : <1 x 1 x i64>
  return %2 : !graphalg.mat<1 x 1 x i64>
}

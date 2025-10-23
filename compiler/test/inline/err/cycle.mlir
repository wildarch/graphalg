// RUN: graphalg-opt %s --graphalg-prepare-inline --verify-diagnostics

// expected-note@below {{Part of the cycle}}
func.func @Cycle0(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply_unary @Cycle1 %arg0 : <1 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}
// expected-note@below {{Part of the cycle}}
func.func @Cycle1(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply_unary @Cycle2 %arg0 : <1 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}
// expected-note@below {{Part of the cycle}}
func.func @Cycle2(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply_unary @Cycle3 %arg0 : <1 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}
// expected-error@below {{The program contains a cycle}}
// expected-note@below {{Part of the cycle}}
func.func @Cycle3(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply_unary @Cycle0 %arg0 : <1 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

func.func @NotPartOfCycle(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  return %arg0 : !graphalg.mat<1 x 1 x i64>
}

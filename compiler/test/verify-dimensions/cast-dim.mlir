// RUN: graphalg-opt --graphalg-verify-dimensions --split-input-file --verify-diagnostics < %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Ok(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.cast_dim #dim
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// -----
#dim = #graphalg.dim<distinct[0]<>>

func.func @IllegalInput() -> !graphalg.mat<1 x 1 x i64> {
  // expected-error@below{{'graphalg.cast_dim' op attribute "input" has value #graphalg.dim<distinct[0]<>> which has not been marked as legal}}
  %0 = graphalg.cast_dim #dim
  return %0 : !graphalg.mat<1 x 1 x i64>
}

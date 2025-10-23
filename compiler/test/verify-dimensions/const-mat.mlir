// RUN: graphalg-opt --graphalg-verify-dimensions --split-input-file --verify-diagnostics < %s
#dim = #graphalg.dim<distinct[0]<>>

func.func @Ok(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
  %0 = graphalg.const_mat 42 : i64 -> <#dim x 1 x i64>
  return %0 : !graphalg.mat<#dim x 1 x i64>
}

// -----
#dim = #graphalg.dim<distinct[0]<>>

func.func @IllegalResultType() -> !graphalg.mat<#dim x 1 x i64> {
  // expected-error@below{{'graphalg.const_mat' op defines type '!graphalg.mat<distinct[0]<> x 1 x i64>' using dimension #graphalg.dim<distinct[0]<>> which has not been marked as legal}}
  %0 = graphalg.const_mat 42 : i64 -> <#dim x 1 x i64>
  return %0 : !graphalg.mat<#dim x 1 x i64>
}

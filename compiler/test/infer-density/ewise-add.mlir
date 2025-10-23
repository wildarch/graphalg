// RUN: graphalg-opt --test-print-dense --verify-diagnostics %s

#dim = #graphalg.dim<distinct[0]<>>
func.func @EwiseAdd(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
  %0 = graphalg.const_mat 0 : i64 -> <#dim x 1 x i64>
  // expected-remark@below {{add}}
  // expected-note@below {{operand #0: unknown}}
  // expected-note@below {{operand #1: dense}}
  // expected-note@below {{result #0: dense}}
  %1 = graphalg.ewise_add %arg0, %0 : !graphalg.mat<#dim x 1 x i64> { tag = "add" }
  return %1 : !graphalg.mat<#dim x 1 x i64>
}

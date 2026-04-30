// RUN: graphalg-opt --test-print-dense --verify-diagnostics %s

#dim = #graphalg.dim<distinct[0]<>>
func.func @For() -> !graphalg.mat<#dim x 1 x f64> {
  %0 = graphalg.const_mat 42.0 : f64 -> <#dim x 1 x f64>
  // expected-remark@below {{for}}
  // expected-note@below {{operand #0: dense}}
  // expected-note@below {{result #0: dense}}
  %1 = graphalg.for begin=0 iters=<10> init(%0) : !graphalg.mat<#dim x 1 x f64>
    -> !graphalg.mat<#dim x 1 x f64> {tag = "for" } body {
  // expected-note@below {{arg #0:0:0: dense}}
  // expected-note@below {{arg #0:0:1: dense}}
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<#dim x 1 x f64>):
    %2 = graphalg.const_mat 42.0 : f64 -> <#dim x 1 x f64>
    graphalg.yield %2 : !graphalg.mat<#dim x 1 x f64>
  } until {
  }
  return %1 : !graphalg.mat<#dim x 1 x f64>
}

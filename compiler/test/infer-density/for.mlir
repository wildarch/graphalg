// RUN: graphalg-opt --test-print-dense --verify-diagnostics %s

#dim = #graphalg.dim<distinct[0]<>>
func.func @ForConst() -> !graphalg.mat<#dim x 1 x f64> {
  %0 = graphalg.const_mat 0 : i64 -> <1 x 1 x i64>
  %1 = graphalg.const_mat 10 : i64 -> <1 x 1 x i64>
  %2 = graphalg.const_mat 42.0 : f64 -> <#dim x 1 x f64>
  // expected-remark@below {{for}}
  // expected-note@below {{operand #0: dense}}
  // expected-note@below {{operand #1: dense}}
  // expected-note@below {{operand #2: dense}}
  // expected-note@below {{result #0: dense}}
  %3 = graphalg.for_const
    range(%0, %1) : <1 x 1 x i64>
    init(%2) : !graphalg.mat<#dim x 1 x f64>
    -> !graphalg.mat<#dim x 1 x f64> {tag = "for" } body {
  // expected-note@below {{arg #0:0:0: dense}}
  // expected-note@below {{arg #0:0:1: dense}}
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<#dim x 1 x f64>):
    %4 = graphalg.const_mat 42.0 : f64 -> <#dim x 1 x f64>
    graphalg.yield %4 : !graphalg.mat<#dim x 1 x f64>
  } until {
  }
  return %3 : !graphalg.mat<#dim x 1 x f64>
}

func.func @ForDim() -> !graphalg.mat<#dim x 1 x f64> {
  %0 = graphalg.const_mat 42.0 : f64 -> <#dim x 1 x f64>
  // expected-remark@below {{for}}
  // expected-note@below {{operand #0: dense}}
  // expected-note@below {{result #0: dense}}
  %1 = graphalg.for_dim range(#dim)
    init(%0) : !graphalg.mat<#dim x 1 x f64>
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

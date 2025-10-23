// RUN: graphalg-opt --graphalg-set-dimensions='func=NoSuchFunction args=42x1' --verify-diagnostics < %s
#dim = #graphalg.dim<distinct[0]<>>

// expected-error@below{{'builtin.module' op does not contain a function named 'NoSuchFunction'}}
module {
  func.func @Func(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
    return %arg0 : !graphalg.mat<#dim x 1 x i64>
  }
}

// RUN: graphalg-opt --graphalg-set-dimensions --verify-diagnostics < %s
#dim = #graphalg.dim<distinct[0]<>>

// expected-error@below {{Missing value for required option 'func'}}
module {
  func.func @Func(%arg0: !graphalg.mat<#dim x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
    return %arg0 : !graphalg.mat<#dim x 1 x i64>
  }
}

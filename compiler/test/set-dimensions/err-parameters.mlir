// RUN: graphalg-opt --graphalg-set-dimensions='func=Func args=4x2,1x1' --split-input-file --verify-diagnostics < %s

// Too few parameters
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

// expected-error@below{{'func.func' op has 1 parameters, expected 2}}
func.func @Func(%arg0: !graphalg.mat<#dim x #dim1 x i64>) -> !graphalg.mat<#dim x #dim1 x i64> {
  return %arg0 : !graphalg.mat<#dim x #dim1 x i64>
}

// -----
// Not matrix type
#dim = #graphalg.dim<distinct[0]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

// expected-error@below {{'func.func' op parameter 1 has type 'i64', expected graphalg.mat}}
func.func @Func(%arg0: !graphalg.mat<#dim x #dim1 x i64>, %arg1: i64) -> !graphalg.mat<#dim x #dim1 x i64> {
  return %arg0 : !graphalg.mat<#dim x #dim1 x i64>
}

// -----
// Concrete dimension not mapped to itself
#dim = #graphalg.dim<distinct[0]<>>

// expected-error@below {{Concrete dimension #graphalg.dim<1> must be mapped to itself, but got 2}}
func.func @Func(%arg0: !graphalg.mat<#dim x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64> {
  return %arg0 : !graphalg.mat<#dim x 1 x i64>
}

// -----
// Abstract dimensions do not match.
#dim = #graphalg.dim<distinct[0]<>>

// expected-error@below {{Attempt to map abstract dimension #graphalg.dim<distinct[0]<>> to concrete dimension 2, but it is already mapped to another concrete dimension 4}}
func.func @Func(%arg0: !graphalg.mat<#dim x #dim x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  return %arg0 : !graphalg.mat<#dim x #dim x i64>
}

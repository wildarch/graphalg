// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

func.func @ConcreteDim() -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: graphalg.const_mat 42 : i64
  %0 = graphalg.cast_dim 42
  return %0 : !graphalg.mat<1 x 1 x i64>
}

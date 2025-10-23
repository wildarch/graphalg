// RUN: graphalg-opt --graphalg-to-core < %s | FileCheck %s

func.func @Lit() -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#CONST:]] = graphalg.const_mat 42 : i64 -> <1 x 1 x i64>
  %0 = graphalg.literal 42 : i64

  // CHECK: return %[[#CONST]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

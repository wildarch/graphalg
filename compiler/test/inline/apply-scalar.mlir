// RUN: graphalg-opt --graphalg-prepare-inline --inline < %s | FileCheck %s
func.func @CallMe(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  return %arg0 : !graphalg.mat<1 x 1 x i64>
}

func.func @Maybe(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: return %arg0
  %0 = graphalg.apply_unary @CallMe %arg0 : <1 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

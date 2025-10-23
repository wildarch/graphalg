// RUN: graphalg-opt --inline < %s | FileCheck %s

func.func @target(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  return %arg0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: func.func @wrapper
func.func @wrapper() -> !graphalg.mat<1 x 1 x i64> {
    // CHECK: %[[#CONST:]] = graphalg.literal 42
    %0 = graphalg.literal 42 : i64
    %1 = func.call @target(%0) : (!graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64>

    // CHECK: return %[[#CONST]]
    return %1 : !graphalg.mat<1 x 1 x i64>
}

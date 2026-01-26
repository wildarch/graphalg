// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

func.func @Tril(%arg0: !graphalg.mat<42 x 42 x i64>) -> !graphalg.mat<42 x 42 x i64> {
  // CHECK: %[[#SELECT:]] = garel.select %arg0
  // CHECK:   %[[#ROW:]] = garel.extract 0
  // CHECK:   %[[#COL:]] = garel.extract 1
  // CHECK:   %[[#CMP:]] = arith.cmpi ult, %[[#COL]], %[[#ROW]]
  // CHECK:   garel.select.return %[[#CMP]]
  %0 = graphalg.tril %arg0 : <42 x 42 x i64>

  // return %[[#SELECT]]
  return %0 : !graphalg.mat<42 x 42 x i64>
}

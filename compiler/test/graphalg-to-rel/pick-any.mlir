// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

func.func @PickAnyMat(%arg0: !graphalg.mat<42 x 43 x i1>) -> !graphalg.mat<42 x 43 x i1> {
  // CHECK: %[[#SELECT:]] = garel.select %arg0
  // CHECK:   %[[#VAL:]] = garel.extract 2
  // CHECK:   %[[#CMP:]] = arith.cmpi ne, %[[#VAL]], %false
  // CHECK:   garel.select.return %[[#CMP]]
  // CHECK: %[[#AGG:]] = garel.aggregate %[[#SELECT]] : <index, index, i1> group_by=[0] aggregators=[<MIN 1>, <ARGMIN 2, 1>]
  %0 = graphalg.pick_any %arg0 : <42 x 43 x i1>

  // CHECK: return %[[#AGG]]
  return %0 : !graphalg.mat<42 x 43 x i1>
}

func.func @PickAnyRowVec(%arg0: !graphalg.mat<1 x 42 x i1>) -> !graphalg.mat<1 x 42 x i1> {
  // CHECK: %[[#SELECT:]] = garel.select %arg0
  // CHECK:   %[[#VAL:]] = garel.extract 1
  // CHECK:   %[[#CMP:]] = arith.cmpi ne, %[[#VAL:]], %false
  // CHECK:   garel.select.return %[[#CMP]]
  // CHECK: %[[#AGG:]] = garel.aggregate %[[#SELECT]] : <index, i1> group_by=[] aggregators=[<MIN 0>, <ARGMIN 1, 0>]
  %0 = graphalg.pick_any %arg0 : <1 x 42 x i1>

  // CHECK: return %[[#AGG]]
  return %0 : !graphalg.mat<1 x 42 x i1>
}

// Note: scalar and column vector cases are folded away

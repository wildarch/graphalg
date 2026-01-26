// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @CastBoolInt
func.func @CastBoolInt(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i1> -> <i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i64> {
  ^bb0(%arg1 : i1):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[C1:.+]] = arith.constant 1 : i64
    // CHECK: %[[C0:.+]] = arith.constant 0 : i64
    // CHECK: %[[#SELECT:]] = arith.select %[[#EXTRACT]], %[[C1]], %[[C0]]
    %1 = graphalg.cast_scalar %arg1 : i1 -> i64

    // CHECK: garel.project.return %[[#SELECT]]
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @CastIntReal
func.func @CastIntReal(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x f64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i64> -> <f64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x f64> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[#CAST:]] = arith.sitofp %[[#EXTRACT]]
    %1 = graphalg.cast_scalar %arg1 : i64 -> f64

    // CHECK: garel.project.return %[[#CAST]]
    graphalg.apply.return %1 : f64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: @CastRealInt
func.func @CastRealInt(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <f64> -> <i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x f64> -> <1 x 1 x i64> {
  ^bb0(%arg1 : f64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[#CAST:]] = arith.fptosi %[[#EXTRACT]]
    %1 = graphalg.cast_scalar %arg1 : f64 -> i64

    // CHECK: garel.project.return %[[#CAST]]
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @CastBoolTrop
func.func @CastBoolTrop(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i1> -> <i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i1> -> <1 x 1 x !graphalg.trop_i64> {
  ^bb0(%arg1 : i1):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[C0:.+]] = arith.constant 0 : i64
    // CHECK: %[[CMAX:.+]] = arith.constant 9223372036854775807 : i64
    // CHECK: %[[#SELECT:]] = arith.select %[[#EXTRACT]], %[[C0]], %[[CMAX]]
    %1 = graphalg.cast_scalar %arg1 : i1 -> !graphalg.trop_i64

    // CHECK: garel.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @CastTropBool
func.func @CastTropBool(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x i1> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i64> -> <i1>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[CMAX:.+]] = arith.constant 9223372036854775807 : i64
    // CHECK: %[[#CMP:]] = arith.cmpi ne, %[[#EXTRACT]], %[[CMAX]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_i64 -> i1

    // CHECK: garel.project.return %[[#CMP]]
    graphalg.apply.return %1 : i1
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @CastTropIntTropReal
func.func @CastTropIntTropReal(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i64> -> <f64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x !graphalg.trop_f64> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[#CAST:]] = arith.sitofp %[[#EXTRACT]]
    // CHECK: %[[MAX:.+]] = arith.constant 9223372036854775807 : i64
    // CHECK: %[[INF:.+]] = arith.constant 0x7FF0000000000000 : f64
    // CHECK: %[[#CMP:]] = arith.cmpi eq, %[[#EXTRACT]], %[[MAX]]
    // CHECK: %[[#SELECT:]] = arith.select %[[#CMP]], %[[INF]], %[[#CAST]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_i64 -> !graphalg.trop_f64

    // CHECK: garel.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_f64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: @CastTropRealTropInt
func.func @CastTropRealTropInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <f64> -> <i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x !graphalg.trop_i64> {
  ^bb0(%arg1 : !graphalg.trop_f64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[#CAST:]] = arith.fptosi %[[#EXTRACT]]
    // CHECK: %[[INF:.+]] = arith.constant 0x7FF0000000000000 : f64
    // CHECK: %[[MAX:.+]] = arith.constant 9223372036854775807 : i64
    // CHECK: %[[#CMP:]] = arith.cmpf oeq, %[[#EXTRACT]], %[[INF]]
    // CHECK: %[[#SELECT:]] = arith.select %[[#CMP]], %[[MAX]], %[[#CAST]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_f64 -> !graphalg.trop_i64

    // CHECK: garel.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @CastIntToTropMaxInt
func.func @CastIntToTropMaxInt(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i64> -> <i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x !graphalg.trop_max_i64> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[C0:.+]] = arith.constant 0 : i64
    // CHECK: %[[MIN:.+]] = arith.constant -9223372036854775808 : i64
    // CHECK: %[[#CMP:]] = arith.cmpi eq, %[[#EXTRACT]], %[[C0]]
    // CHECK: %[[#SELECT:]] = arith.select %[[#CMP]], %[[MIN]], %[[#EXTRACT]]
    %1 = graphalg.cast_scalar %arg1 : i64 -> !graphalg.trop_max_i64

    // CHECK: garel.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_max_i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

// CHECK-LABEL: @CastTropMaxIntToInt
func.func @CastTropMaxIntToInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#PROJECT:]] = garel.project %arg0 : <i64> -> <i64>
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> -> <1 x 1 x i64> {
  ^bb0(%arg1 : !graphalg.trop_max_i64):
    // CHECK: %[[#EXTRACT:]] = garel.extract 0
    // CHECK: %[[MIN:.+]] = arith.constant -9223372036854775808 : i64
    // CHECK: %[[C0:.+]] = arith.constant 0 : i64
    // CHECK: %[[#CMP:]] = arith.cmpi eq, %[[#EXTRACT]], %[[MIN]]
    // CHECK: %[[#SELECT:]] = arith.select %[[#CMP]], %[[C0]], %[[#EXTRACT]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_max_i64 -> i64

    // CHECK: garel.project.return %[[#SELECT]]
    graphalg.apply.return %1 : i64
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i64>
}

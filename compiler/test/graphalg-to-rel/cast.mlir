// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// CHECK-LABEL: @CastBoolInt
func.func @CastBoolInt(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i64> {
  ^bb0(%arg1 : i1):
    // CHECK: %[[#SLOT:]] = ipr.slot
    // CHECK: %[[#MUL_IDENT:]] = ipr.constant_slot <<"S64"> (1)>
    // CHECK: %[[#ADD_IDENT:]] = ipr.constant_slot <<"S64"> (0)>
    // CHECK: %[[#SELECT:]] = ipr.select_slot %[[#SLOT]]
    // CHECK-SAME: [<<"BOOLEAN"> (false)>] %[[#ADD_IDENT]]
    // CHECK-SAME: default = %[[#MUL_IDENT]]
    %1 = graphalg.cast_scalar %arg1 : i1 -> i64

    // CHECK: ipr.project.return %[[#SELECT]]
    graphalg.apply.return %1 : i64
  }

  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @CastIntReal
func.func @CastIntReal(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x f64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x f64> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[#INPUT:]] = ipr.slot
    // CHECK: %[[#CAST:]] = ipr.cast %[[#INPUT]] : si64 -> <"F64">
    %1 = graphalg.cast_scalar %arg1 : i64 -> f64

    // CHECK: ipr.project.return %[[#CAST]]
    graphalg.apply.return %1 : f64
  }

  return %0 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: @CastRealInt
func.func @CastRealInt(%arg0: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x f64> -> <1 x 1 x i64> {
  ^bb0(%arg1 : f64):
    // CHECK: %[[#INPUT:]] = ipr.slot
    // CHECK: %[[#CAST:]] = ipr.cast %[[#INPUT]] : f64 -> <"S64">
    %1 = graphalg.cast_scalar %arg1 : f64 -> i64

    // CHECK: ipr.project.return %[[#CAST]]
    graphalg.apply.return %1 : i64
  }

  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @CastBoolTrop
func.func @CastBoolTrop(%arg0: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i1> -> <1 x 1 x !graphalg.trop_i64> {
  ^bb0(%arg1 : i1):
    // CHECK: %[[#SLOT:]] = ipr.slot

    // CHECK: %[[#MUL_IDENT:]] = ipr.constant_slot <<"S64"> (0)>
    // CHECK: %[[#ADD_IDENT:]] = ipr.constant_slot <<"S64"> (9223372036854775807)>
    // CHECK: %[[#SELECT:]] = ipr.select_slot %[[#SLOT]]
    // CHECK-SAME: [<<"BOOLEAN"> (false)>] %[[#ADD_IDENT]]
    // CHECK-SAME: default = %[[#MUL_IDENT]]
    %1 = graphalg.cast_scalar %arg1 : i1 -> !graphalg.trop_i64

    // CHECK: ipr.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_i64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @CastTropBool
func.func @CastTropBool(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x i1> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: %[[#INPUT:]] = ipr.slot
    // CHECK: %[[#ZERO:]] = ipr.constant_slot <<"S64"> (9223372036854775807)>
    // CHECK: %[[#CMP:]] = ipr.cmp %[[#INPUT]] : si64 NE %[[#ZERO]] : si64
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_i64 -> i1

    // CHECK: ipr.project.return %[[#CMP]]
    graphalg.apply.return %1 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @CastTropIntTropReal
func.func @CastTropIntTropReal(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64> -> <1 x 1 x !graphalg.trop_f64> {
  ^bb0(%arg1 : !graphalg.trop_i64):
    // CHECK: %[[#INPUT:]] = ipr.slot
    // CHECK: %[[#CAST:]] = ipr.cast %[[#INPUT]] : si64 -> <"F64">
    //
    // CHECK: %[[#INF:]] = ipr.constant_slot <<"F64"> (INF)>
    //
    // CHECK: %[[#SELECT:]] = ipr.select_slot %[[#INPUT]]
    // CHECK-SAME: [<<"S64"> (9223372036854775807)>]
    // CHECK-SAME: %[[#INF]]
    // CHECK-SAME: default = %[[#CAST]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_i64 -> !graphalg.trop_f64

    // CHECK: ipr.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_f64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: @CastTropRealTropInt
func.func @CastTropRealTropInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64> -> <1 x 1 x !graphalg.trop_i64> {
  ^bb0(%arg1 : !graphalg.trop_f64):
    // CHECK: %[[#INPUT:]] = ipr.slot
    // CHECK: %[[#CAST:]] = ipr.cast %[[#INPUT]] : f64 -> <"S64">
    //
    // CHECK: %[[#INF:]] = ipr.constant_slot <<"S64"> (9223372036854775807)>
    //
    // CHECK: %[[#SELECT:]] = ipr.select_slot %[[#INPUT]]
    // CHECK-SAME: [<<"F64"> (INF)>]
    // CHECK-SAME: %[[#INF]]
    // CHECK-SAME: default = %[[#CAST]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_f64 -> !graphalg.trop_i64

    // CHECK: ipr.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_i64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @CastIntToTropMaxInt
func.func @CastIntToTropMaxInt(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x !graphalg.trop_max_i64> {
  ^bb0(%arg1 : i64):
    // CHECK: %[[#INPUT:]] = ipr.slot

    // CHECK: %[[#ZERO:]] = ipr.constant_slot <<"S64"> (-9223372036854775808)>
    //
    // CHECK: %[[#SELECT:]] = ipr.select_slot %[[#INPUT]]
    // CHECK-SAME: [<<"S64"> (0)>] %[[#ZERO]]
    // CHECK-SAME: default = %[[#INPUT]]
    %1 = graphalg.cast_scalar %arg1 : i64 -> !graphalg.trop_max_i64

    // CHECK: ipr.project.return %[[#SELECT]]
    graphalg.apply.return %1 : !graphalg.trop_max_i64
  }

  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

// CHECK-LABEL: @CastTropMaxIntToInt
func.func @CastTropMaxIntToInt(%arg0: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x i64> {
  %0 = graphalg.apply %arg0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> -> <1 x 1 x i64> {
  ^bb0(%arg1 : !graphalg.trop_max_i64):
    // CHECK: %[[#INPUT:]] = ipr.slot

    // CHECK: %[[#ZERO:]] = ipr.constant_slot <<"S64"> (0)>
    //
    // CHECK: %[[#SELECT:]] = ipr.select_slot %[[#INPUT]]
    // CHECK-SAME: [<<"S64"> (-9223372036854775808)>] %[[#ZERO]]
    // CHECK-SAME: default = %[[#INPUT]]
    %1 = graphalg.cast_scalar %arg1 : !graphalg.trop_max_i64 -> i64

    // CHECK: ipr.project.return %[[#SELECT]]
    graphalg.apply.return %1 : i64
  }

  return %0 : !graphalg.mat<1 x 1 x i64>
}

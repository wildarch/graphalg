// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

// CHECK-LABEL: func.func @Add
func.func @Add(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.add %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 ADD %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: func.func @SubInt
func.func @SubInt(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = arith.subi %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 SUB %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: func.func @SubReal
func.func @SubReal(%arg0: !graphalg.mat<#dim x #dim x f64>) -> !graphalg.mat<#dim x #dim x f64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = arith.subf %arg1, %arg1 : f64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x f64> -> <#dim x #dim x f64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x f64>):
    %1 = graphalg.ewise %arg1 SUB %arg1 : <1 x 1 x f64>
    graphalg.apply_inline.return %1 : <1 x 1 x f64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x f64>
}

// CHECK-LABEL: func.func @Mul
func.func @Mul(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.mul %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 MUL %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i64>
}

// CHECK-LABEL: func.func @Div
func.func @Div(%arg0: !graphalg.mat<#dim x #dim x f64>) -> !graphalg.mat<#dim x #dim x f64> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = arith.divf %arg1, %arg1 : f64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x f64> -> <#dim x #dim x f64> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x f64>):
    %1 = graphalg.ewise %arg1 DIV %arg1 : <1 x 1 x f64>
    graphalg.apply_inline.return %1 : <1 x 1 x f64>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x f64>
}

// CHECK-LABEL: func.func @Eq
func.func @Eq(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.eq %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 EQ %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

// CHECK-LABEL: func.func @Ne
func.func @Ne(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#FALSE:]] = graphalg.const false
  // CHECK:   %[[#EQ:]] = graphalg.eq %arg1, %arg1 : i64
  // CHECK:   %[[#RES:]] = graphalg.eq %[[#FALSE]], %[[#EQ]]
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 NE %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

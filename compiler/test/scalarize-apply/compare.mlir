// RUN: graphalg-opt --graphalg-scalarize-apply < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

// CHECK-LABEL: func.func @Lt
func.func @Lt(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.lt %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 LT %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

// CHECK-LABEL: func.func @Gt
func.func @Gt(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.lt %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 GT %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

// CHECK-LABEL: func.func @Le
func.func @Le(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.le %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 LE %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

// CHECK-LABEL: func.func @Ge
func.func @Ge(%arg0: !graphalg.mat<#dim x #dim x i64>) -> !graphalg.mat<#dim x #dim x i1> {
  // CHECK: %[[#APPLY:]] = graphalg.apply %arg0
  // CHECK:   %[[#RES:]] = graphalg.le %arg1, %arg1 : i64
  // CHECK:   graphalg.apply.return %[[#RES]]
  %0 = graphalg.apply_inline %arg0 : !graphalg.mat<#dim x #dim x i64> -> <#dim x #dim x i1> {
  ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>):
    %1 = graphalg.ewise %arg1 GE %arg1 : <1 x 1 x i64>
    graphalg.apply_inline.return %1 : <1 x 1 x i1>
  }

  // CHECK: return %[[#APPLY]]
  return %0 : !graphalg.mat<#dim x #dim x i1>
}

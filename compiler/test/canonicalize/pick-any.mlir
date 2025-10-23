// RUN: graphalg-opt --canonicalize < %s | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>
!scalar_i64 = !graphalg.mat<1 x 1 x i64>
!vec_i64 = !graphalg.mat<#dim x 1 x i64>
!vec_i1 = !graphalg.mat<#dim x 1 x i1>
!mat_i1 = !graphalg.mat<#dim x #dim x i1>

// CHECK-LABEL: @PickAnyScalar
func.func @PickAnyScalar(%arg0: !scalar_i64) -> !scalar_i64 {
  // CHECK: return %arg0
  %0 = graphalg.pick_any %arg0 : !scalar_i64
  return %0 : !scalar_i64
}

// CHECK-LABEL: @PickAnyVec
func.func @PickAnyVec(%arg0: !vec_i64) -> !vec_i64 {
  // CHECK: return %arg0
  %0 = graphalg.pick_any %arg0 : !vec_i64
  return %0 : !vec_i64
}

// CHECK-LABEL: @PickAnyReduce
func.func @PickAnyReduce(%arg0: !mat_i1) -> !mat_i1 {
  // CHECK: %[[#PICK:]] = graphalg.pick_any %arg0
  %0 = graphalg.deferred_reduce %arg0 : !mat_i1 -> !mat_i1
  %1 = graphalg.pick_any %0 : !mat_i1

  // return %[[#PICK]]
  return %1 : !mat_i1
}

// Check that our rewrite rules do not eliminate both the deferred_reduce AND
// the pick_any.
// CHECK-LABEL: @PickAnyReduceVec
func.func @PickAnyReduceVec(%arg0: !mat_i1) -> !vec_i1 {
  // CHECK %[[#REDUCE:]] = graphalg.deferred_reduce %arg0
  %0 = graphalg.deferred_reduce %arg0 : !mat_i1 -> !vec_i1
  %1 = graphalg.pick_any %0 : !vec_i1

  // return %[[#REDUCE]]
  return %1 : !vec_i1
}

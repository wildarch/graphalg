// RUN: graphalg-opt --graphalg-to-rel < %s | FileCheck %s

// === Input/output shapes

// CHECK-LABEL: @ReduceMatScalar
func.func @ReduceMatScalar(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, index, i64> group_by=[] aggregators=[<SUM 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ReduceMatRowVec
func.func @ReduceMatRowVec(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<1 x 43 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, index, i64> group_by=[1] aggregators=[<SUM 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x i64> -> <1 x 43 x i64>
  return %0 : !graphalg.mat<1 x 43 x i64>
}

// CHECK-LABEL: @ReduceMatColVec
func.func @ReduceMatColVec(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, index, i64> group_by=[0] aggregators=[<SUM 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x i64> -> <42 x 1 x i64>
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// CHECK-LABEL: @ReduceMatMat
func.func @ReduceMatMat(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<42 x 43 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, index, i64> group_by=[0, 1] aggregators=[<SUM 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x i64> -> <42 x 43 x i64>
  return %0 : !graphalg.mat<42 x 43 x i64>
}

// CHECK-LABEL: @ReduceRowVecScalar
func.func @ReduceRowVecScalar(%arg0: !graphalg.mat<1 x 43 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, i64> group_by=[] aggregators=[<SUM 1>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<1 x 43 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ReduceRowVecRowVec
func.func @ReduceRowVecRowVec(%arg0: !graphalg.mat<1 x 43 x i64>) -> !graphalg.mat<1 x 43 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, i64> group_by=[0] aggregators=[<SUM 1>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<1 x 43 x i64> -> <1 x 43 x i64>
  return %0 : !graphalg.mat<1 x 43 x i64>
}

// CHECK-LABEL: @ReduceColVecScalar
func.func @ReduceColVecScalar(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, i64> group_by=[] aggregators=[<SUM 1>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ReduceColVecColVec
func.func @ReduceColVecColVec(%arg0: !graphalg.mat<42 x 1 x i64>) -> !graphalg.mat<42 x 1 x i64> {
  // CHECK: garel.aggregate %arg0 : <index, i64> group_by=[0] aggregators=[<SUM 1>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 1 x i64> -> <42 x 1 x i64>
  return %0 : !graphalg.mat<42 x 1 x i64>
}

// CHECK-LABEL: @ReduceScalarScalar
func.func @ReduceScalarScalar(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: garel.aggregate %arg0 : <i64> group_by=[] aggregators=[<SUM 0>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<1 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ReduceMultiple
func.func @ReduceMultiple(
    %arg0 : !graphalg.mat<1 x 43 x i64>,
    %arg1 : !graphalg.mat<42 x 1 x i64>)
    -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: %[[#UNION:]] = garel.union %arg0, %arg1 : !garel.rel<index, i64>, !garel.rel<index, i64>
  // CHECK: %[[#AGG:]] = garel.aggregate %0 : <index, i64> group_by=[] aggregators=[<SUM 1>]
  %0 = graphalg.deferred_reduce %arg0, %arg1 : !graphalg.mat<1 x 43 x i64>, !graphalg.mat<42 x 1 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// === Semirings

// CHECK-LABEL: @ReduceBool
func.func @ReduceBool(%arg0: !graphalg.mat<42 x 43 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  // CHECK: aggregators=[<LOR 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x i1> -> <1 x 1 x i1>
  return %0 : !graphalg.mat<1 x 1 x i1>
}

// CHECK-LABEL: @ReduceInt
func.func @ReduceInt(%arg0: !graphalg.mat<42 x 43 x i64>) -> !graphalg.mat<1 x 1 x i64> {
  // CHECK: aggregators=[<SUM 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x i64> -> <1 x 1 x i64>
  return %0 : !graphalg.mat<1 x 1 x i64>
}

// CHECK-LABEL: @ReduceReal
func.func @ReduceReal(%arg0: !graphalg.mat<42 x 43 x f64>) -> !graphalg.mat<1 x 1 x f64> {
  // CHECK: aggregators=[<SUM 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x f64> -> <1 x 1 x f64>
  return %0 : !graphalg.mat<1 x 1 x f64>
}

// CHECK-LABEL: @ReduceTropInt
func.func @ReduceTropInt(%arg0: !graphalg.mat<42 x 43 x !graphalg.trop_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_i64> {
  // CHECK: aggregators=[<MIN 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x !graphalg.trop_i64> -> <1 x 1 x !graphalg.trop_i64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_i64>
}

// CHECK-LABEL: @ReduceTropReal
func.func @ReduceTropReal(%arg0: !graphalg.mat<42 x 43 x !graphalg.trop_f64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_f64> {
  // CHECK: aggregators=[<MIN 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x !graphalg.trop_f64> -> <1 x 1 x !graphalg.trop_f64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_f64>
}

// CHECK-LABEL: @ReduceTropMaxInt
func.func @ReduceTropMaxInt(%arg0: !graphalg.mat<42 x 43 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x !graphalg.trop_max_i64> {
  // CHECK: aggregators=[<MAX 2>]
  %0 = graphalg.deferred_reduce %arg0 : !graphalg.mat<42 x 43 x !graphalg.trop_max_i64> -> <1 x 1 x !graphalg.trop_max_i64>
  return %0 : !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>
}

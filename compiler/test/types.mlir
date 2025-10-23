// RUN: graphalg-opt < %s | FileCheck %s

// CHECK: #dim = #graphalg.dim<distinct[0]<>>
#dim = #graphalg.dim<distinct[0]<>>
//CHECK: #dim1 = #graphalg.dim<distinct[1]<>>
#dim1 = #graphalg.dim<distinct[1]<>>

!scalar_i1 = !graphalg.mat<1 x 1 x i1>
!colvec_i64 = !graphalg.mat<#dim x 1 x i64>
!rowvec_f64 = !graphalg.mat<1 x #dim1 x f64>
!matrix_t64 = !graphalg.mat<#dim x #dim1 x !graphalg.trop_i64>
!concrete = !graphalg.mat<42 x 43 x i64>

// CHECK-LABEL: test_scalar_i1
// CHECK: !graphalg.mat<1 x 1 x i1>
func.func @test_scalar_i1(%0 : !scalar_i1) -> !scalar_i1 {
    return %0 : !scalar_i1
}

// CHECK-LABEL: test_colvec_i64
// CHECK: !graphalg.mat<#dim x 1 x i64>
func.func @test_colvec_i64(%0 : !colvec_i64) -> !colvec_i64 {
    return %0 : !colvec_i64
}

// CHECK-LABEL: test_rowvec_f64
// CHECK: !graphalg.mat<1 x #dim1 x f64>
func.func @test_rowvec_f64(%0 : !rowvec_f64) -> !rowvec_f64 {
    return %0 : !rowvec_f64
}

// CHECK-LABEL: test_matrix_t64
// CHECK: !graphalg.mat<#dim x #dim1 x !graphalg.trop_i64>
func.func @test_matrix_t64(%0 : !matrix_t64) -> !matrix_t64 {
    return %0 : !matrix_t64
}

// CHECK-LABEL: test_concrete
// CHECK: !graphalg.mat<42 x 43 x i64>
func.func @test_concrete(%0 : !concrete) -> !concrete {
    return %0 : !concrete
}

// RUN: mlir-opt -allow-unregistered-dialect < %s
// RUN: graphalg-opt -verify-roundtrip -allow-unregistered-dialect < %s | FileCheck %s
// RUN: graphalg-opt -emit-bytecode -allow-unregistered-dialect < %s | graphalg-opt -allow-unregistered-dialect | FileCheck %s

#distinct = distinct[0]<[]>
#distinct1 = distinct[1]<[]>
// CHECK: #dim = #graphalg.dim<#distinct>
#dim = #graphalg.dim<#distinct>
// CHECK: #dim1 = #graphalg.dim<#distinct1>
#dim1 = #graphalg.dim<#distinct1>

// CHECK-LABEL: @TropInf
module @TropInf attributes {
    // CHECK: test.f64 = #graphalg.trop_inf : !graphalg.trop_f64,
    test.f64 = #graphalg.trop_inf : !graphalg.trop_f64,
    // CHECK: test.i64 = #graphalg.trop_inf : !graphalg.trop_i64
    test.i64 = #graphalg.trop_inf : !graphalg.trop_i64
} {}

module @DimAttr attributes {
    // CHECK: test.concrete = #graphalg.dim<42>
    test.concrete = #graphalg.dim<42>,
    // CHECK: test.dim0 = #dim,
    test.dim0 = #dim,
    // CHECK: test.dim1 = #dim
    test.dim1 = #dim1,
    // CHECK: test.one = #graphalg.dim<1>
    test.one = #graphalg.dim<1>
} {}

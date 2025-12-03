# Conversion to GraphBLAS
Documents how to convert GraphAlg Core to GraphBLAS operations.

Conversion is done in two stages:
1. GraphAlg Core to GraphBLAS with value semantics. We can do additional high-level optimization here.
2. value to by reference semantics. Effectively this is bufferization.

## Semirings
- `bool`: `GrB_LOR_LAND_SEMIRING_BOOL`
- `int`: `GrB_PLUS_TIMES_SEMIRING_INT64`
- `real`: `GrB_PLUS_TIMES_SEMIRING_FP64`
- `trop_int`: `GrB_MIN_PLUS_SEMIRING_INT64`
- `trop_real`: `GrB_MIN_PLUS_SEMIRING_FP64`
- `trop_max_int`: `GrB_MAX_PLUS_SEMIRING_INT64`

## Sparsity
The simplest approach is to reuse `MakeDenseOp`.

## Accumulation
GraphBLAS operations support fused mask/accumulation via flags passed to each operation.

NOTE: `GrB_Matrix_diag` does not have mask/accumulate flags.

## Matrix Operations

### `TransposeOp`
Use `GrB_transpose`. Note that various GraphBLAS ops allow you set a flag for inputs that should be transposed, so some canonicalization/folding would be useful here.

### `DiagOp`
Use `GrB_Matrix_diag`.

### `MatMulOp`
Use `GrB_mxm`, `GrB_vxm` or `GrB_mxv` depending on the argument types.

### `ReduceOp`
Use `GrB_reduce`.

### `BroadcastOp`
Use `GrB_assign`.

### `ConstantMatrixOp`
Use `GrB_assign` with `GrB_ALL` for row and column indices.

### `ForConstOp`
Interpreter handles this.

### `ApplyOp`
Convert `ApplyOp` bodies into element-wise operations.

Operations that need to be converted:
- `ConstantOp` -> `ConstantMatrixOp`
- `AddOp` -> `GrB_eWiseMult` with op from semiring
- `MulOp` -> `GrB_eWiseMult` with op from semiring
- `CastScalarOp` -> Custom user-defined functions.
- `EqOp` -> `GrB_eWiseMult` with `GrB_EQ_T`
- `DivIOp` -> `GrB_eWiseMult` with `GrB_DIV_INT64`
- `DivFOp` -> `GrB_eWiseMult` with `GrB_DIV_FP64`
- `SubIOp` -> `GrB_eWiseMult` with `GrB_MINUS_INT64`
- `SubFOp` -> `GrB_eWiseMult` with `GrB_MINUS_FP64`

### `PickAnyOp`
Implemented as a custom C++ routine.

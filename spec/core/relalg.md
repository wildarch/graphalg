---
title: Conversion to Relational Algebra
layout: page
parent: Core Language
nav_order: 3
---

# Conversion to Relational Algebra
TODO:
- Data model for the conversion
- Definition of the assumed loop operator
- Definition of the assumed aggregation operation
- Rewrite rules for the conversion

## Data Model
The target system/algebra is assumed to support the following data types:
- Boolean, denoted `i1`
- 64-bit signed integer, denoted `si64`
- 64-bit floating-point (IEEE 754 is assumed not strictly required), denoted `f64`

## Functions
The conversion applies to individual functions, i.e. we can convert a single function call (with relational algebra expressions for the parameters) into a relational algebra expression.
This is sufficient because as opposed to the full GraphAlg language, in GraphAlg Core functions do not call other functions.
The `return` expression in the body becomes the root of the resulting expression.

## Decomposing MatMul and ReduceOp
TODO: Describe how these ops are broken up into `MatMulJoinOp`, `UnionOp` and `DeferredReduceOp`.

## Matrix Expressions

### `TransposeOp`
Translates into a projection that swaps the row and column indices.

```
projection(
    <expr>,
    {
        row = col,
        col = row,
        val = val,
    }
)
```

TODO: Could be a vector

### `DiagOp`
Translates into a projection that adds a column index equal to the row index.

```
projection(
    <expr>,
    {
        row = row,
        col = row,
        val = val,
    }
)
```

TODO: Could be a row vector too.

### `BroadcastOp`
For each dimension added, we join with a constant table of all possible indices for that dimension.

### `ForConstOp`
Translates directly to the relational algebra loop definition.
For loops that produce multiple outputs, we duplicate the loop.

TODO: Need to define the loop structure before we can give the translation.

### `PickAnyOp`
Remove zero elements, then select zero (col,val) combinations per row:

```
aggregate(
    selection(
        <expr>,
        { val != <zero> }
    ),
    group by { row },
    aggregate { min(col, val) }
)
```

TODO: for boolean matrices, `min` with a single input is sufficient, because `val` is always `true`.

### `ApplyOp`
- Join all inputs based on available columns (some inputs may not have all dimensions and need to be broadcast)
- If none of the inputs provides a requested output column, add additional joins to constant tables to broadcast them
- Create a projection with the body of the `ApplyOp`. Keep only one row/col slot (as necessary for the output).

### `TrilOp`
Keep all tuples with `col < row`:

```
selection(
    <expr>,
    { col < row }
)
```

### `ConstantMatrixOp`
Create a constant table.

```
(row, col, val)
(0, 0, 42)
(0, 1, 42)
(0, 2, 42)
...
(1, 0, 42)
(1, 1, 42)
(1, 2, 42)
...
```

### `MatMulJoinOp`
- Join the matrices on `<lhs>.col = <rhs>.row`
- Project out the multiplied values

### `DeferredReduceOp`
Aggregate, grouping by row/col, merging values according to the semiring add operator (see `AddOp` conversion).

### `UnionOp`
Simple relational union.

## Scalar Expressions
### `ApplyReturnOp`
Becomes return op inside projection.

### `ConstantOp`
Becomes a constant in a plain data type depending on the semiring:
- `i1` and `f64` are kept as-is
- `i64` becomes `si64`
- `trop_int` becomes `si64`. If the constant value is infinity, we map this to the the maximum value of the `si64` type (i.e. the largest possible integer)
- `trop_max_int` becomes `si64`. If the constant value is infinity, we map this to the the minimum value of the `si64` type.
- `trop_real` becomes `f64`. In IEEE 754 `f64` has a proper infinity value, so we can map `trop_real` infinity to that.

### `CastScalarOp`
Cases:
- `* -> i1`: rewrite to `input != zero(inRing)`
- `i1 -> *`: rewrite to `input ? one(outRing) : zero(outRing)`
- `int -> real`: integer promotion
- `int -> trop_int`: map to infinity (max value) if 0, otherwise leave unchanged.
- `int -> trop_max_int`: map to infinity if 0, otherwise leave unchanged.
- `int -> trop_real`: map to infinity if 0, otherwise promote
- `real -> int`: truncate
- `real -> trop_int`: map to infinity if 0, otherwise promote
- `real -> trop_max_int`: map to infinity if 0, otherwise promote.
- `real -> trop_real`: map to infinity if 0, otherwise leave unchanged.

### `AddOp`
Pick the operation based on the semiring:
- `i1`: logical OR
- `i64`: signed integer add
- `f64`: floating point add
- `trop_i64`/`trop_real`: `min`
- `trop_max_i64`: `max`

### `mlir::arith::SubIOp`
Keep as-is.

### `mlir::arith::SubFOp`
Keep as-is.

### `MulOp`
Pick the operation based on the semiring:
- `i1`: logical AND
- `i64`: signed integer multiply
- `f64`: floating point multiply
- `trop_i64`/`trop_max_i64`: signed integer add
- `trop_real`: floating point add

### `mlir::arith::DivFOp`
Keep as-is.

### `EqOp`
Keep as-is.

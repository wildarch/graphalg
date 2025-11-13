---
title: Conversion to Relational Algebra
layout: page
parent: Core Language
nav_order: 3
---

# Conversion to Relational Algebra
This page describes how GraphAlg Core operations can be converted into an extended relational algebra.

{: .note }
Conversion to relational algebra requires that all matrices use concrete types rather than abstract dimension symbols.
The `graphalg-set-dimensions` pass can be used to set concrete values for dimension symbols.

## Data Model
The target system/algebra is assumed to support the following data types:
- Boolean, denoted `i1`
- 64-bit signed integer, denoted `i64`
- 64-bit floating-point (IEEE 754 is assumed not strictly required), denoted `f64`

It must support the following standard relational algebra operators:
- projection: Drops or reorders input columns, or computes new columns based on existing ones using per-tuple operations.
- selection: Keeps a subset of the input tuples based on a per-tuple predicate.
- join: Combines two or more relations.

Additionally, an operator to perform aggregation is required, which is not strictly part of relational algebra, but is commonly available in relational database systems.
The `aggregate` operator must support grouping tuples by key columns, and it must support the following aggregator functions to combine values:
- `sum`, `min` and `max` over `i64` and `f64`
- `or` over `i1` (representing logical OR)
- `argmin(arg, val)`, which finds the tuple with minimal `val` and produces the `arg` for that tuple.

A last requirement, and one unusual to relational database systems, is a loop operator.
Its semantics are similar to `ForConstOp`.

## Loop Operator
The loop operator repeatedly evaluates a collection of relational algebra expressions based on *loop variables* whose state can change between iterations. The initial state of the loop variable is provided as inputs to the loop operator in the form of

The loop operator has a number of inputs:
- initial states for the loop variables, provided as relational algebra expressions
- a maximum number of iterations, given as an integer constant
- a result index, indicating which of the loop variables will become the final output of the loop

The loop operator embeds *loop body expressions*, one relational algebra expression per loop variable, which together represent the loop body.
Besides all the usual query operators, a loop body expression can reference loop variables to read the current state of a variable.

Optionally included is a *break expression*, a relational algebra expression that indicates if the loop should terminate early, before completing the maximum number of iterations. This expression should produce tuples with a single boolean value, where `true` in one of the tuple indicates that the loop should terminate.

To run one iteration of the loop, first the loop body expressions are evaluated based on the current state of the loop variables.
Then, the values of the loop variables are replaced with the results of loop body expressions.
This process continue until the maximum number of iterations has been reached, or until the *break expression* returns a tuple with value to `true`.

## Functions
The conversion applies to individual functions, i.e. a single function call (with relational algebra expressions for the parameters) converts into a relational algebra expression.
This is sufficient because as opposed to the full GraphAlg language, in GraphAlg Core functions do not call other functions.
The `return` expression in the body becomes the root of the resulting expression.

## Decomposing `MatMulOp` and `ReduceOp`
`MatMulOp` and `ReduceOp` are replaced with `MatMulJoinOp`, `UnionOp` and `DeferredReduceOp` in the `graphalg-split-aggregate` pass.
`MatMulOp` is decomposed into `MatMulJoinOp + DeferredReduceOp`, whereas `ReduceOp` becomes `UnionOp + DeferredReduceOp`.
For this reason, only the translation of `MatMulJoinOp`, `UnionOp` and `DeferredReduceOp` is considered.

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

For vector inputs that lack either `row` or `col` columns, transpose is a no-op.

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

If the input is a row vector rather than a column vector, a row index equal to the column index is added instead.

### `BroadcastOp`
For each dimension added, join with a constant table of all possible indices for that dimension.

### `ForConstOp`
Translates directly to the relational algebra loop definition.
For loops that produce multiple outputs, the loop is duplicated.
Because all operations are deterministic, this does not change the semantics of the program.

### `PickAnyOp`
Remove zero elements, then select zero (col,val) combinations per row:

```
aggregate(
    selection(
        <expr>,
        { val != <zero> }
    ),
    group by { row },
    aggregate {
        min(col),
        argmin(val, col)
    }
)
```

{: .note }
For boolean matrices, the value of `argmin(val, col)` is always `true`, so this aggregator can be omitted.

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
For a

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
- `i64` becomes `i64`
- `trop_int` becomes `i64`. If the constant value is infinity, we map this to the the maximum value of the `i64` type (i.e. the largest possible integer)
- `trop_max_int` becomes `i64`. If the constant value is infinity, we map this to the the minimum value of the `i64` type.
- `trop_real` becomes `f64`. In IEEE 754, `f64` has a proper infinity value, so we can map `trop_real` infinity to that.

### `CastScalarOp`
Cases:
- `* -> i1`: rewrite to `input != zero(inRing)`
- `i1 -> *`: rewrite to `input ? one(outRing) : zero(outRing)`
- `i64 -> f64`: integer promotion
- `i64 -> trop_i64`: map to infinity (max value) if 0, otherwise leave unchanged.
- `i64 -> trop_max_i64`: map to infinity if 0, otherwise leave unchanged.
- `i64 -> trop_f64`: map to infinity if 0, otherwise promote
- `f64 -> i64`: truncate
- `f64 -> trop_i64`: map to infinity if 0, otherwise promote
- `f64 -> trop_max_i64`: map to infinity if 0, otherwise promote.
- `f64 -> trop_f64`: map to infinity if 0, otherwise leave unchanged.

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
Convert to the equivalent compare operation in the target representation.

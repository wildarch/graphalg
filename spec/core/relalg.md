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

### `TrilOp`
Keep all tuples with `col < row`:

```
selection(
    <expr>,
    { col < row }
)
```

### `ApplyOp`
- Join all inputs based on available columns (some inputs may not have all dimensions and need to be broadcast)
- If none of the inputs provides a requested output column, add additional joins to constant tables to broadcast them
- Create a projection with the body of the `ApplyOp`. Keep only one row/col slot (as necessary for the output).

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


TODO:
- `TransposeOp`
- `DiagOp`
- `BroadcastOp`
- `ForConstOp`
- `PickAnyOp`
- `TrilOp`
- `ConstantMatrixOp`
- `ApplyOp`
- `MatMulJoinOp`
- `DeferredReduceOp`
- `UnionOp`

## Scalar Expressions
- `ApplyReturnOp`
- `ConstantOp`
- `CastScalarOp`
- `AddOp`
- `mlir::arith::SubIOp`
- `mlir::arith::SubFOp`
- `MulOp`
- `mlir::arith::DivFOp`
- `EqOp`

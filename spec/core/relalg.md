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

## Matrix Expressions
TODO:
- `TransposeOp`
- `DiagOp`
- `BroadcastOp`
- `ForConstOp`
- `PickAnyOp`
- `TrilOp`
- `ConstantMatrixOp`
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

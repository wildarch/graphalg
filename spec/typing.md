---
title: Type System
layout: page
parent: Language Specification
nav_order: 2
---

# Type System
TODO:
- Type enviroment and rule syntax
- The actual type checking rules

## Notation
Functions are typed based on a function environment `F`.
The function environment associates function names to function types, e.g `F(myFunc) = (Matrix<s, s, int>, Matrix<1, 1, real>) -> Matrix<1, 1, int>`.

For statement and expression typing rules we add two additional pieces of context:
- Local environment `L`, a map associating variables to types, i.e. `L(a) = Matrix<s, s, int>`.
- Return type `R`, the expected return type of the enclosing function.

## Functions
<img src="latex/program.svg" width="1024"/>

<img src="latex/function.svg" width="600"/>

## Statements

### Assign
If there is no existing binding for the variable to be assigned, the first rule applies. Otherwise, the more involved reassignment rule applies.

<img src="latex/stmt-assign.svg" width="600"/>

The following additional typing rules deal with updating dimension symbol context $D$ if the expression to be assigned represents a dimension. It also defines the semantics of $\vdash_D$ to infer the dimension represented by an expression.

<img src="latex/dim.svg" width="600"/>

### Accumulate
<img src="latex/stmt-accum.svg" width="600"/>

### For
<img src="latex/stmt-for.svg" width="600"/>

## Expressions

### Variable
<img src="latex/expr-var.svg" width="600"/>

### Transpose
<img src="latex/expr-transpose.svg" width="600"/>

### Matrix Properties (`nrows`, `ncols`, `nvals`)
<img src="latex/expr-prop.svg" width="600"/>

### Scalar Arithmetic
<img src="latex/expr-arith.svg" width="600"/>

### Compare
<img src="latex/expr-compare.svg" width="600"/>

### New Matrix/Vector
<img src="latex/expr-new.svg" width="600"/>

### Cast

### Literal

### Zero/One

### diag

### apply
- unary
- binary

### select
- unary
- binary

### Reduce (rows/cols)

### pickAny

### element-wise

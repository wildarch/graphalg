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

## Statements

## Expressions

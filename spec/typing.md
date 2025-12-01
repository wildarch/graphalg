---
title: Type System
layout: page
parent: Language Specification
nav_order: 3
---

# Type System
We define the GraphAlg type system formally using [typing rules](https://en.wikipedia.org/wiki/Typing_rule).

## Notation

### Environments
Typing rules use three distinct environments:
- Function environment $$F$$: Associates function names with the parameter and results types of those functions.
- Dimension environment $D$: Associates variable names with dimension symbols.
- Local environment $L$: Associates variable names with a matrix type.

The environment objects have map semantics:
- $M \in L$ asserts that $L$ contains a value for $M$.
- $L[M]$ retrieves the value for key $M$ from environment $L$.
  Implies $M \in L$.
- $L' := L[M := \tau]$ creates an updated environment $L'$ that contains all mappings from $L$ plus a mapping from $M$ to $\tau$.
  If $M \in L$, then the original value for $M$ is dropped ($L'[M] = \tau$).

### Type Instances
Type instances are denoted $\langle d_1, d_2, R \rangle$, where
- $d_1$ is the dimension symbol for the number of rows
- $d_2$ is the dimension symbol for the number of columns
- $R$ is the semiring

### Bindings
We use the syntax $A := B$ to bind the value of $B$ to a new variable $A$.
This binding syntax is also used to destructure types, e.g. $\langle d_1, d_2, R \rangle := \tau$ allows referencing the dimension symbols and the semiring of $\tau$. Furthermore, the syntax can also be used for asserting (partial) matches. For example, the statement $\langle d, 1, R_1 \rangle = \tau_1$ following by $\langle 1, d, R_2 \rangle$ asserts that:
- $\tau_1$ has one column
- $\tau_2$ has one row
- The number of rows in $\tau_1$ matches the number of columns in $\tau_2$

No requirement is placed on the relation of $R_1$ and $R_2$: they may refer to the same semirings, or to two different ones.

## Functions
Functions have a function type of the form $(\tau_1 \times \tau_2) \rightarrow \tau_r$.
A program is nothing more than a collection of functions.
Note that the order in which functions are defined is significant:
A function cannot be referenced before they are defined.

<img src="latex/program.svg" width="600"/>

<img src="latex/function.svg" width="600"/>

## Statements
Statements update the $D$ and $L$ environments.
They do not have a type of their own.

### Assign
If there is no existing binding for the variable to be assigned, the first rule applies.
Otherwise, the more involved reassignment rule applies.

<img src="latex/stmt-assign.svg" width="600"/>

Complementing the mask does not influence the typing in any way.
It is omitted from the typing rule for simplicity.

The following additional typing rules deal with updating dimension symbol context $D$ if the expression to be assigned represents a dimension. It also defines the semantics of $\vdash_D$ to infer the dimension represented by an expression.

<img src="latex/dim.svg" width="600"/>

### Accumulate
The variable to accumulate into must be defined.
The expression to accumulate must have the same type.

<img src="latex/stmt-accum.svg" width="600"/>

### For
We give one rule for loops with an integer bound (start and end), and another for loops over a dimension symbol.

<img src="latex/stmt-for.svg" width="600"/>

Additional variables defined inside the loop scope are not available outside that scope, except for in the `until` expression.

## Expressions
Well-typed expressions have an associated matrix type $\tau$ as determined by the following rules.

### Variable
Variables must be defined before they are referenced.

<img src="latex/expr-var.svg" width="600"/>

### Matrix Multiply
Matrix multiplication requires that the number of columns on the left-hand side matches the number of rows on the right-hand side.

<img src="latex/expr-matmul.svg" width="600"/>

### New Matrix/Vector
Arguments to matrix/vector creation must refer to dimension symbols.

<img src="latex/expr-new.svg" width="600"/>

### Transpose
Transpose is valid for any input matrix.

<img src="latex/expr-transpose.svg" width="600"/>

### `diag`
`diag` requires the input to be either a row or a column vector.

<img src="latex/expr-diag.svg" width="600"/>

### `apply`
Functions used in `apply` must be defined over scalar input and output types.
The parameter semirings must match those of the respective input arguments.

<img src="latex/expr-apply.svg" width="600"/>

### `select`
The typing is similar to `apply`, but the result of `f` should be a boolean.

<img src="latex/expr-select.svg" width="600"/>

### `reduce`
All variants of `reduce` are valid for any input matrix.

<img src="latex/expr-reduce.svg" width="600"/>

### `pickAny`
`pickAny` does not change the type of the input matrix.

<img src="latex/expr-pickAny.svg" width="600"/>

### Element-wise Operations
Input types for the left and right-hand sides must match, except for function application.

<img src="latex/expr-ewise.svg" width="600"/>

### Cast
All conversions between semirings are allowed.

<img src="latex/expr-cast.svg" width="600"/>

### Matrix Properties (`nrows`, `ncols`, `nvals`)
All matrix properties return an integer.

<img src="latex/expr-prop.svg" width="600"/>

### Scalar Arithmetic
Arithmetic operations are defined for scalar (1 by 1) inputs.
Operations such as subtraction or division are only allowed on specific semirings.

<img src="latex/expr-arith.svg" width="600"/>

### Scalar Compare
Ordered comparison operations are only allowed on the `int` and `real` semirings.
Equality comparison is allowed regardless of the semiring.

<img src="latex/expr-compare.svg" width="600"/>

### Literal
Literals must be valid for the specified semiring.

<img src="latex/expr-literal.svg" width="600"/>

<script>
MathJax = {
  tex: {
    inlineMath: {'[+]': [['$', '$']]}
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"></script>

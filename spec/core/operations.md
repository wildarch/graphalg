---
title: Operations
layout: page
parent: Core Language
nav_order: 1
---

# Operations in GraphAlg Core
This page defines the operational semantics of the operations in GraphAlg Core.
Operations are referred to by their MLIR Op name.
GraphAlg Core is a compiler-internal IR that has no defined syntax apart from the standard MLIR syntax.

{:.warning-title}
> Under Construction
>
> This part of the documentation is not yet finished.

## Matrix Operations

### `TransposeOp`
Matrix Transpose.

```math
O_{ij} = I_{ji}
```

### `DiagOp`
Diagonalizes the input vector into matrix.

If the input is a column vector:
```math
O_{ii} = I_{0,i}
```

If the input is row vector:
```math
O_{ii} = I_{i,0}
```

### `MatMulOp`
Matrix multiplication using the addition operator and multiplication operator as defined by the matrix semiring.

```math
O_{ik} = \bigoplus_j L_{ij} \otimes R_{jk}
```

### `ReduceOp`
Reduce input matrix along one or two dimensions.
If the output type is scalar:

```math
O_{0,0} = \bigoplus_{ij} I_{ij}
```

If the output type is a column vector:

```math
O_{i,0} = \bigoplus_{j} I_{ij}
```

If the output type is row vector:

```math
O_{0,j} = \bigoplus_{i} I_{ij}
```

### `BroadcastOp`
Broadcast the input to a higher dimension by duplicating elements.

TODO: Define the semantics in a concise way (use 1-vector?)

### `ConstantMatrixOp`
Fill a matrix with a constant scalar value.

```math
O_{ij} = c
```

### `ForConstOp`
TODO: Define precise semantics.

### `ApplyOp`
TODO: Describe op semantics.

TODO: Note implicit broadcasting

### `PickAnyOp`
TODO: Describe op semantics.

## Scalar Operations
### `ConstantOp`
Scalar constant.

```math
O = c
```

NOTE: Implementation also needs to handle `mlir::arith::ConstantOp` due to folding of `arith` ops.

### `AddOp`
Applies the semiring addition operator.

```math
O = l \oplus r
```

### `MulOp`
Applies the semiring multiplication operator.

```math
O = l \otimes r
```

### `CastScalarOp`
TODO: Describe op semantics.

### `EqOp`
Tests whether two scalars have the same value.

```math
O = (l = r)
```

### `mlir::arith::DivFOp`
Floating point division. Dividing by zero is well-defined, producing a zero-value output.

```math
O = \begin{cases}
    0 & \text{ if } r = 0 \\
    l / r & \text{ otherwise }
\end{cases}
```

### `mlir::arith::SubIOp`/`mlir::arith::SubFOp`
Integer or floating-point subtraction.

```math
O = l - r
```

<script src="{{ '/playground/editor.bundle.js' | relative_url }}"></script>

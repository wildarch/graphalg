# Parser Testing Guide

This guide explains how to write and test parser error tests for the GraphAlg compiler.

## Test Framework

The GraphAlg compiler uses **LLVM's `lit` (LLVM Integrated Tester)** framework for testing. Tests are located in `compiler/test/`.

### Test Types

1. **Success tests** (`compiler/test/parse/`) - Verify correct parsing and MLIR output using FileCheck
2. **Error tests** (`compiler/test/parse-err/`) - Verify that invalid code produces expected error messages

## Writing Parser Error Tests

### Basic Structure

```graphalg
// RUN: graphalg-translate --import-graphalg --verify-diagnostics %s

func MyTest() -> int {
    // expected-error@below{{error message}}
    invalid code here;
    return int(0);
}
```

### Key Components

1. **RUN directive**: Specifies how to run the test
   - `--verify-diagnostics` flag enables diagnostic verification
   - `%s` is replaced with the test file path

2. **Diagnostic annotations**: Tell the verifier what errors/notes to expect
   - `// expected-error@below{...}` - expects an error on the next line
   - `// expected-note@below{...}` - expects a note/additional info on the next line
   - Messages in braces must match the actual diagnostic message exactly

3. **Error location**: The verifier checks that errors appear at the expected location

### Multiple Diagnostics

When multiple diagnostics appear on the same statement, list them all:

```graphalg
// expected-error@below{{base dimensions do not match the dimensions of the mask}}
// expected-note@below{{base dimension: (s x s)}}
// expected-note@below{{mask dimensions: (t x t)}}
a<m> = e;
```

## Running Tests

### Run All Tests

```bash
cmake --build compiler/build --target check
```

### Run Individual Tests

```bash
# First build the compiler
cmake --build compiler/build --target graphalg-translate

# Run with diagnostic verification
./compiler/build/tools/graphalg-translate --import-graphalg --verify-diagnostics path/to/test.gr

# Run without verification to see actual errors
./compiler/build/tools/graphalg-translate --import-graphalg path/to/test.gr
```

## Common Error Categories

### 1. Duplicate Definitions

**Function names** (`func-name-dup.gr`):
```graphalg
// expected-note@below{{original definition here}}
func Dup(a: int) -> int { return a; }

// expected-error@below{{duplicate definition of function 'Dup'}}
func Dup(a: int) -> int { return a; }
```

**Parameter names** (`func-param-dup.gr`):
```graphalg
func Dup(
    // expected-note@below{{previous definition here}}
    a: int,
    // expected-error@below{{duplicate parameter name 'a'}}
    a: int) -> int { return a; }
```

### 2. Fill Syntax Errors

**Vector fill on non-vector** (`vector-fill-row-vector.gr`):
```graphalg
func Test(m: Matrix<1, s, int>, x: int) -> Matrix<1, s, int> {
    // expected-error@below{{vector fill [:] used with non-vector base}}
    // expected-note@below{{base has type Matrix<1, s, int>}}
    m[:] = x;
    return m;
}
```

**Matrix fill on vector** (`matrix-fill-col-vector.gr`):
```graphalg
func Test(v: Vector<s, int>, x: int) -> Vector<s, int> {
    // expected-error@below{{matrix fill [:, :] used with column vector base}}
    // expected-note@below{{base has type Vector<s, int>}}
    v[:, :] = x;
    return v;
}
```

**Non-scalar fill expression** (`vector-fill-non-scalar.gr`):
```graphalg
func Test(v: Vector<s, int>, e: Vector<s, int>) -> Vector<s, int> {
    // expected-error@below{{fill expression is not a scalar}}
    v[:] = e;
    return v;
}
```

### 3. Masked Assignment Errors

**Dimension mismatch** (`mask-dimension-mismatch.gr`):
```graphalg
func Test(a: Matrix<s, s, int>, m: Matrix<t, t, bool>, e: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{base dimensions do not match the dimensions of the mask}}
    // expected-note@below{{base dimension: (s x s)}}
    // expected-note@below{{mask dimensions: (t x t)}}
    a<m> = e;
    return a;
}
```

### 4. Type Errors

**Reassignment with different type** (`reassign-type-mismatch.gr`):
```graphalg
func Test() -> int {
    // expected-note@below{{previous assigment was here}}
    a = int(42);
    // expected-error@below{{cannot assign value of type real to previously defined variable of type int}}
    a = real(3.14);
    return int(0);
}
```

**Accumulate type mismatch** (`accum-type-mismatch.gr`):
```graphalg
func Test() -> int {
    a = int(42);
    // expected-error@below{{type of base does not match the expression to accumulate: (int vs. real}}
    a += real(3.14);
    return int(0);
}
```

### 5. Variable Scoping

**Undefined variable** (`accum-undefined.gr`):
```graphalg
func Test() -> int {
    // expected-error@below{{undefined variable}}
    a += int(42);
    return int(0);
}
```

**Loop scope** (`loop-scope.gr`):
```graphalg
func Test() -> int {
    a = int(0);
    for i in int(1):int(10) {
        b = int(42);
        a = a + b;
    }
    // Variable b is not accessible outside the loop
    // expected-error@below{{unrecognized variable}}
    return b;
}
```

### 6. For Loop Errors

**Non-integer range bounds** (`loop-range-start-non-int.gr`, `loop-range-end-non-int.gr`):
```graphalg
func Test() -> int {
    a = int(0);
    // expected-error@below{{loop range start must be an integer, but got real}}
    for i in real(1.0):int(10) {
        a = a + int(1);
    }
    return a;
}
```

**Non-dimension range** (`loop-range-not-dimension.gr`):
```graphalg
func Test(m: Matrix<s, s, int>) -> int {
    a = int(0);
    // expected-error@below{{not a dimension type}}
    // expected-note@below{{defined here}}
    for i in m.nvals {
        a = a + int(1);
    }
    return a;
}
```

**Non-boolean until condition** (`loop-until-non-bool.gr`):
```graphalg
func Test() -> int {
    a = int(0);
    for i in int(1):int(10) {
        a = a + int(1);
    // expected-error@below{{loop condition does not produce a boolean scalar, got int}}
    } until int(5);
    return a;
}
```

## Type Formatting in Error Messages

The parser formats types in a user-friendly way:
- Scalars: `int`, `real`, `bool`, `trop_int`, `trop_real`
- Vectors: `Vector<s, int>` (column vector with dimension `s` and element type `int`)
- Matrices: `Matrix<r, c, int>` (matrix with row dimension `r`, column dimension `c`, and element type `int`)

## Tips for Writing Tests

1. **Test one error at a time** - Keep tests focused on a single error condition
2. **Use descriptive function names** - Name the function after what it tests
3. **Add comments** - Explain what the test is checking
4. **Verify exact error messages** - The diagnostic message must match exactly
5. **Check location precision** - Ensure the error points to the right token
6. **Test boundary cases** - Cover edge cases like row vectors, column vectors, and full matrices

## Adding Parser Error Checks

When adding new error detection to the parser:

1. **Identify the error condition** in the parser code
2. **Choose an appropriate error message** - Be clear and user-friendly
3. **Use `mlir::emitError(location)`** to report the error
4. **Add notes with `diag.attachNote()`** for additional context
5. **Write a test** that verifies the error is caught
6. **Run the test** to verify the exact error message format
7. **Update the test** with the correct expected message

### Example: Adding Loop Range Type Checks

```cpp
// Check that begin is an integer scalar
auto intScalarType = MatrixType::scalarOf(SemiringTypes::forInt(_builder.getContext()));
if (r.begin.getType() != intScalarType) {
  return mlir::emitError(beginLoc)
         << "loop range start must be an integer, but got "
         << typeToString(r.begin.getType());
}
```

### 7. Return Statement Errors

**Return inside loop** (`return-in-loop.gr`):
```graphalg
func Test() -> int {
    for i in int(1):int(10) {
        // expected-error@below{{return statement inside a loop is not allowed}}
        return int(5);
    }
    return int(0);
}
```

**Return not last statement** (`return-not-last.gr`):
```graphalg
func Test() -> int {
    return int(42);
    // expected-error@below{{statement after return is not allowed}}
    a = int(5);
    return a;
}
```

**Return wrong type** (`return-wrong-type.gr`):
```graphalg
func Test() -> int {
    // expected-error@below{{return type mismatch: expected int, but got real}}
    return real(3.14);
}
```

**Multiple return statements** (`return-multiple.gr`):
```graphalg
func Test() -> int {
    a = int(42);
    return a;
    // expected-error@below{{statement after return is not allowed}}
    return int(5);
}
```

**Missing return statement** (`return-missing.gr`):
```graphalg
// expected-error@below{{function must have a return statement}}
func Test() -> int {
    a = int(42);
}
```

### 8. Matrix Multiplication Errors

**Dimension mismatch** (`matmul-dimension-mismatch.gr`):
```graphalg
func MatMulDimensionMismatch(
    // expected-note@below{{left side has dimensions (r x s)}}
    a: Matrix<r, s, int>,
    // expected-note@below{{right side has dimensions (t x u)}}
    b: Matrix<t, u, int>) -> Matrix<r, u, int> {
    // expected-error@below{{incompatible dimensions for matrix multiply}}
    return a * b;
}
```

### 9. Built-in Function Errors

**diag() with non-vector** (`diag-not-vector.gr`):
```graphalg
func DiagNotVector(
    // expected-note@below{{argument has type Matrix<r, c, int>}}
    m: Matrix<r, c, int>) -> Matrix<r, r, int> {
    // expected-error@below{{diag() requires a row or column vector}}
    return diag(m);
}
```

**apply() with wrong function signature** (`apply-unary-func-wrong-arg-count.gr`):
```graphalg
// expected-note@below{{function defined here}}
func twoArgs(a: int, b: int) -> int {
    return a + b;
}

func ApplyUnaryFuncWrongArgCount(m: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{apply() with 1 matrix argument requires a function with 1 parameter, but got 2}}
    return apply(twoArgs, m);
}
```

**apply() with non-scalar function parameters** (`apply-func-non-scalar-args.gr`):
```graphalg
// expected-note@below{{parameter 0 has type Matrix<s, s, int>}}
func nonScalarFunc(m: Matrix<s, s, int>) -> int {
    return int(0);
}

func ApplyFuncNonScalarArgs(m: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{apply() requires function parameters to be scalars}}
    return apply(nonScalarFunc, m);
}
```

**select() with non-bool return type** (`select-func-non-bool-return.gr`):
```graphalg
// expected-note@below{{function returns int}}
func returnsInt(a: int) -> int {
    return a;
}

func SelectFuncNonBoolReturn(m: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{select() requires function to return bool}}
    return select(returnsInt, m);
}
```

### 10. Element-wise Operation Errors

**Different semirings** (`ewise-different-semirings.gr`):
```graphalg
func EwiseDifferentSemirings(
    // expected-note@below{{left operand has semiring int}}
    a: Matrix<s, s, int>,
    // expected-note@below{{right operand has semiring real}}
    b: Matrix<s, s, real>) -> Matrix<s, s, int> {
    // expected-error@below{{element-wise operation requires operands to have the same semiring}}
    return a (.+) b;
}
```

**Different dimensions** (`ewise-different-dimensions.gr`):
```graphalg
func EwiseDifferentDimensions(
    // expected-note@below{{left operand has dimensions (s x s)}}
    a: Matrix<s, s, int>,
    // expected-note@below{{right operand has dimensions (t x t)}}
    b: Matrix<t, t, int>) -> Matrix<s, s, int> {
    // expected-error@below{{element-wise operation requires operands to have the same dimensions}}
    return a (.+) b;
}
```

### 11. Subtraction and Negation Errors

**Subtraction with unsupported semiring** (`sub-bool-unsupported.gr`, `sub-trop-int-unsupported.gr`):
```graphalg
func SubBoolUnsupported(
    // expected-note@below{{operands have semiring bool}}
    a: bool, b: bool) -> bool {
    // expected-error@below{{subtraction is only supported for int and real types}}
    return a - b;
}
```

**Subtraction with non-scalars** (`sub-matrix-not-scalar.gr`):
```graphalg
func SubMatrixNotScalar(
    // expected-note@below{{left operand has type Matrix<s, s, int>}}
    a: Matrix<s, s, int>,
    // expected-note@below{{right operand has type Matrix<s, s, int>}}
    b: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // Note: Use element-wise subtraction (.-) for matrices
    // expected-error@below{{subtraction only works on scalars; use element-wise subtraction (.-) for matrices}}
    return a - b;
}
```

**Negation with unsupported semiring** (`neg-bool-unsupported.gr`, `neg-trop-int-unsupported.gr`):
```graphalg
func NegBoolUnsupported(
    // expected-note@below{{operand has semiring bool}}
    a: bool) -> bool {
    // expected-error@below{{negation is only supported for int and real types}}
    return -a;
}
```

## Current Test Coverage

As of now, we have 173 parser tests covering:
- Duplicate definitions (functions and parameters)
- Fill syntax errors (vector vs matrix, non-scalar expressions)
- Masked assignment errors (dimension mismatches)
- Variable reassignment errors (type mismatches)
- Accumulate errors (undefined variable, type mismatches)
- Variable scoping (loop-local variables)
- For loop errors (non-integer start/end, non-dimension range, non-boolean until condition)
- Return statement errors (in loop, not last, wrong type, multiple returns, missing return)
- Matrix multiplication dimension mismatches
- Built-in function validation (diag, apply, select)
- Element-wise operation type checking (semiring and dimension compatibility)
- Subtraction and negation semiring restrictions (only int and real)
- **Comparison operator semiring restrictions (only int, real, and trop_real)**
- **Division semiring restrictions (only real and trop_int)**
- **NOT operator type restrictions (only bool)**
- **Literal type conversion validation**
- **Element-wise function application validation (parameter count, types, and existence)**
### 12. Comparison Operator Errors

**Comparison with unsupported semiring** (`cmp-bool-unsupported.gr`, `cmp-trop-int-unsupported.gr`):
```graphalg
func CmpBoolUnsupported(
    // expected-note@below{{operands have semiring bool}}
    a: bool, b: bool) -> bool {
    // expected-error@below{{comparison is only supported for int, real, and trop_real types}}
    return a < b;
}
```

Comparison operators (`<`, `>`, `<=`, `>=`) only support int, real, and trop_real semirings. They are not supported for bool or trop_int.

### 13. Division Errors

**Division with unsupported semiring** (`div-int-unsupported.gr`, `div-trop-real-unsupported.gr`, `div-bool-unsupported.gr`):
```graphalg
func DivIntUnsupported(
    // expected-note@below{{operands have semiring int}}
    a: int, b: int) -> int {
    // expected-error@below{{division is only supported for real and trop_int types}}
    return a / b;
}
```

Division only supports real and trop_int semirings. It is not supported for int, bool, or trop_real.

### 14. NOT Operator Errors

**NOT with non-bool semiring** (`not-int-unsupported.gr`, `not-real-unsupported.gr`, `not-trop-int-unsupported.gr`):
```graphalg
func NotIntUnsupported(
    // expected-note@below{{operand has semiring int}}
    a: int) -> int {
    // expected-error@below{{not operator is only supported for bool type}}
    return !a;
}
```

The NOT operator (`!`) only works with bool type. It is not supported for any other semiring.

### 15. Literal Type Conversion Errors

**Invalid literal conversions** (`literal-bool-from-int.gr`, `literal-int-from-real.gr`, `literal-trop-real-from-bool.gr`):
```graphalg
func LiteralBoolFromInt() -> bool {
    // expected-error@below{{expected 'true' or 'false'}}
    return bool(42);
}

func LiteralIntFromReal() -> int {
    // expected-error@below{{expected an integer value}}
    return int(42.0);
}

func LiteralTropRealFromBool() -> trop_real {
    // expected-error@below{{expected a floating-point value}}
    return trop_real(false);
}
```

Each semiring type requires a specific literal format:
- `bool` requires `true` or `false`
- `int` and `trop_int` require integer literals
- `real` and `trop_real` require floating-point literals

### 16. Element-wise Function Application Errors

Element-wise function application uses the syntax `a (.func) b` to apply a binary function element-wise to two matrices.

**Wrong parameter count** (`ewise-func-wrong-param-count.gr`):
```graphalg
// expected-note@below{{function defined here}}
func oneParam(a: int) -> int {
    return a;
}

func Test(m: Matrix<s, s, int>, n: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{element-wise function application requires a function with 2 parameters, but got 1}}
    return m (.oneParam) n;
}
```

**Non-scalar parameters** (`ewise-func-non-scalar-params.gr`):
```graphalg
// expected-note@below{{parameter 0 has type Matrix<s, s, int>}}
func matrixParam(m: Matrix<s, s, int>, n: int) -> int {
    return n;
}

func Test(m: Matrix<s, s, int>, n: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{element-wise function application requires function parameters to be scalars}}
    return m (.matrixParam) n;
}
```

**Type mismatch** (`ewise-func-type-mismatch-left.gr`, `ewise-func-type-mismatch-right.gr`):
```graphalg
// expected-note@below{{first parameter has type real}}
func addReals(a: real, b: real) -> real {
    return a + b;
}

func Test(
    // expected-note@below{{left operand has type int}}
    m: Matrix<s, s, int>, n: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{left operand type does not match first parameter type}}
    return m (.addReals) n;
}
```

**Undefined function** (`ewise-func-undefined.gr`):
```graphalg
func Test(m: Matrix<s, s, int>, n: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // expected-error@below{{unknown function 'doesNotExist'}}
    return m (.doesNotExist) n;
}
```

## Best Practices for Error Messages

### Always Use TypeFormatter for Type Display

When emitting error messages that include types, **always use `typeToString()`** instead of directly printing MLIR types. This ensures user-friendly error messages.

**Correct:**
```cpp
auto diag = mlir::emitError(loc)
            << "parameter has type " << typeToString(funcType.getInput(i));
```

**Incorrect:**
```cpp
auto diag = mlir::emitError(loc)
            << "parameter has type " << funcType.getInput(i);  // Shows raw MLIR type
```

The `typeToString()` function uses `TypeFormatter` internally, which formats types in a user-friendly way:
- Raw MLIR: `!graphalg.mat<distinct[0]<> x distinct[0]<> x i64>`
- Formatted: `Matrix<s, s, int>`

### Semiring Type Restrictions Summary

| Operation | Supported Semirings | Notes |
|-----------|-------------------|-------|
| Addition (`+`) | All | No restrictions |
| Subtraction (`-`) | int, real | Scalar only without `(.-)` |
| Multiplication (`*`) | All | Matrix multiplication has dimension requirements |
| Division (`/`) | real, trop_int | |
| Comparison (`<`, `>`, `<=`, `>=`) | int, real, trop_real | |
| Equality (`==`, `!=`) | All | No restrictions |
| NOT (`!`) | bool | Only boolean values |
| Negation (`-x`) | int, real | Unary operator |

### Validation Checklist for Element-wise Function Application

When implementing validation for element-wise function application `a (.func) b`:

1. ✅ Function must exist (checked by `parseFuncRef`)
2. ✅ Function must have exactly 2 parameters
3. ✅ All function parameters must be scalars
4. ✅ Left operand type must match first parameter type
5. ✅ Right operand type must match second parameter type

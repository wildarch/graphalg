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

### Duplicate Definitions

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

### Type Errors

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

### Variable Scoping

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

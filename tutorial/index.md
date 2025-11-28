---
title: Tutorial
layout: page
nav_order: 2
---

# Tutorial

## Welcome
Welcome to the GraphAlg tutorial.
GraphAlg is a domain-specific programming language for writing graph algorithms.
As you will soon see, with GraphAlg you can use familiar linear algebra operations such as matrix multiplication to analyze graphs.
GraphAlg is designed to be embedded into database systems, allowing you to run complex user-defined graph algorithms without leaving your DBMS.

This guide is designed for new users that want to learn how to write graph algorithms in GraphAlg.
To follow along, all you need are:
1. Basic programming skills in another language (Python, JavaScript, etc.)
2. Rudimentary knowledge of linear algebra (Matrix multiplication).

## A First Example
Let us start with an example program to introduce key concepts of the GraphAlg language.
Try running it by pressing the **Run** button!.

{: .note }
> The examples shown in this tutorial run inside your own browser using
> [WebAssembly](https://webassembly.org/).
>
> You can modify and run examples as much as you want.
> The name of the function to execute appears in the *Run* button above the editor.
> To see the values of the parameters passed to the function, click on the *Argument* accordions below the editor to expand them.
>
> If you write an invalid program, the editor will underline the part of your
> program that is incorrect. Hover over the underlined code to see the error
> message, or click *Run* to show the error messages below the editor.

{:
    data-ga-func="AddOne"
    data-ga-arg-0="
        1, 1, i64;
        0, 0, 42;"
}
```graphalg
func AddOne(a: int) -> int {
    return a + int(1);
}
```

As you may have guessed, this trivial program simply increments its input by one.
Or, more accurately, the `AddOne` function increments its input by one.
A GraphAlg program is nothing more than a collection of functions.

## Anatomy of a Function
Functions are defined using the `func` keyword.
GraphAlg is *statically typed*: you must assign types to the parameters of a function, and to the return type.
The general shape of a function is:

```graphalg
func <function name>(<param name>: <param type>, ...) -> <return type> {
    <statement>
    ...
}
```

The `AddOne` uses the `int` type for both the parameter and the return type.
This represents a signed, 64-bit integer (similar to `long` in Java and C++).
A function can have any number of parameters, but it must have **exactly one return value**.
You may be familiar with other languages that allow returning no value at all (commonly named `void` functions).
In GraphAlg such a function would be pointless:
GraphAlg programs cannot write to files, make HTTP requests, or perform any other action that has [*side effects*](https://en.wikipedia.org/wiki/Side_effect_(computer_science)).
A GraphAlg function can only perform a computation over the inputs it receives, and return the result.

{: .note-title }
> Why no side effects?
>
> Disallowing side effects may seem like an annoying restriction to place on a competent programmer such as yourself, but it is crucial for systems that implement GraphAlg support:
> By not allowing side effects, GraphAlg programs can be heavily optimized to run as efficiently as possible.
> It also makes it possible to implement GraphAlg in highly diverse and restrictive environments, such as the query engine of a database system.

Coming back to the shape a function, let us consider the body of the function (the part contained inside `{..}`).
The body consists of one or more statements that together define the behaviour of the function.
A function body ends with a `return` statement that defines the final result to be returned.

## More Statements: Variables and Loops
Our `AddOne` function was simple enough that we could directly define the result inside of the `return`.
Let us now consider a more complex program that needs a few more statements:

{:
    data-ga-func="Fibonacci"
    data-ga-arg-0="
        1, 1, i64;
        0, 0, 10;"
}
```graphalg
func Fibonacci(n: int) -> int {
    a = int(0);
    b = int(1);
    for i in int(0):n {
        c = a + b;
        a = b;
        b = c;
    }

    return b;
}
```

As the name implies, `Fibonacci` computes the `n`'th number in the fibonacci sequence.
This example shows how to define new variables with `=` (see line 2, where we define `a`).
The same syntax is used to update the value of an existing variable (see line 6, where we reassign `a`).

{: .note-title }
> Experiment
>
> Try computing different numbers in the sequence by adding e.g. `n = int(5);` at the start of the function body.

We also see a first use of the `for` construct. Like many other programming languages, `for` executes its loop body repeatedly.
Variable `i` is the *loop iteration variable*.
It is defined by loop, and assigned to the current iteration number.
The values that `i` will take are defined by the *loop range*, `int(0):n` in the example above.
Assuming you have not changed the default value of `n` (10), the loop will run for 10 iterations, where `i` takes on the values 0, 1, 2, 3, 4, 5, 6, 7, 8 and finally 9 **(not 10)**.

A word about the scope of variables:
You can refer to variables defined earlier in the same scope (`{ .. }` defines a scope), or to variables that were defined *before* entering the current scope.
For example, `c = a + b;` on line 5 can refer to `a` and `b` even though they are defined in the outer function scope, not the loop scope.
This is okay because the loop scope is defined *inside* of the function scope.
What is not allowed however is to first define a variable in a nested scope, and then refer to it from the outer scope.
For an example, consider the (invalid) program below.

{:
    data-ga-func="NotValid"
}
```graphalg
func NotValid() -> int {
    a = int(0);
    for i in int(0):int(10) {
        b = a;
        a = a + int(1);
    }

    return b;
}
```

Variable `b` is only defined inside of the loop, not in the outer function scope.
The statement `return b;` is invalid, as `b` is undefined in this context.

{: .note-title }
> Experiment
>
> Fix the compiler error by defining `b` at the function scope before entering the loop.

## Bringing Linear Algebra Into the Mix
Our example programs so far have only used the `int` type so far.
There are also `bool` (`false` or `true`) and `real` (floating-point numbers, 64-bit).
To capture the connected nature of graphs, however, we need more than simple scalar types.
GraphAlg represents graphs as adjacency matrices.
Consider the graph shown below:

{:
    data-ga-mode="vis"
}
```graphalg-matrix
4, 4, i1;
0, 1;
0, 2;
1, 3;
2, 3;
```

In adjacency matrix representation, the same graph looks like this:

```graphalg-matrix
4, 4, i1;
0, 1;
0, 2;
1, 3;
2, 3;
```

In the matrix representation we can use linear algebra operations to explore the graph.
Want to find all reachable nodes from node 0?
We can do that with matrix multiplication.

First, we create an initial vector with value 1 at position 0, and multiply that with the adjacency matrix to get all nodes that are reachable from node 0 in a single hop:

```math
\begin{bmatrix}
1 & 0 & 0 & 0 \\
\end{bmatrix}

\cdot

\begin{bmatrix}
0 & 1 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}

=

\begin{bmatrix}
0 & 1 & 1 & 0
\end{bmatrix}
```

So nodes 1 and 2 are one hop away from node 0. How about two hops?

```math
\begin{bmatrix}
1 & 0 & 0 & 0 \\
\end{bmatrix}

\cdot

\begin{bmatrix}
0 & 1 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}

\cdot

\begin{bmatrix}
0 & 1 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}

=

\begin{bmatrix}
0 & 1 & 1 & 0
\end{bmatrix}

\cdot

\begin{bmatrix}
0 & 1 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}

=

\begin{bmatrix}
0 & 0 & 0 & 1
\end{bmatrix}
```

So node 3 is also reachable.
All nodes in this graph are reachable from node 0!

## A First Graph Algorithm
Let us now codify this strategy for finding reachable nodes in the graph by writing a GraphAlg program.
If you run the program below (click the *Run 'Reachability'* button), you will see that it finds four nodes that are reachable from node 0.
Two additional nodes 4 and 5 are not reachable from node 0.

{:
    data-ga-func="Reachability"
    data-ga-arg-0="
6, 6, i1;
0, 1;
0, 2;
1, 3;
2, 3;
4, 5;"
    data-ga-arg-1="6, 1, i1; 0, 0;"
    data-ga-result-render="vertex-property"
}
```graphalg
func Reachability(
        graph: Matrix<s, s, bool>,
        source: Vector<s, bool>) -> Vector<s, bool> {
    reach = source;
    for i in graph.nrows {
        reach += reach * graph;
    }

    return reach;
}
```

The reachability algorithm uses a few new GraphAlg features that we have not encountered so far:
- More complex types `Matrix<..>` and `Vector<..>` (line 2-3)
- Iterating over the dimensions of a matrix (line 5)
- Accumulating results using `+=` (line 6)
- Matrix multiplication using `*` (line 6)

## Matrix Types
A `Matrix` type encodes three properties:
1. The number of rows.
   This example uses a *symbolic name* `s` rather than a concrete value, so that we can apply the algorithm to matrices of any size.
2. The number of columns.
   In the example this is also `s`, so parameter `graph` is a square matrix.
3. The *semiring*.
   You can see this as the type of elements in the matrix, for example `int`.
   We will give a more precise definition further on in the tutorial.

`Vector<s, bool>` is an alias for `Matrix<s, 1, bool>`, a column vector.

{: .note-title}
> A note on scalar types
>
> Even the simple scalar types you have seen before, such as `int`, are matrices!
> `int` is a shorthand for `Matrix<1, 1, int>`.

## Loops over Matrix Dimensions
We have previously seen loops over an integer range such as `for i in int(0):int(10) {..}`.
In graph algorithms it is very common to bound loops based on the number of vertices in the graph, so GraphAlg provides shorthand syntax `for i in graph.nrows {..}`.
It is equivalent to `for i in int(0):graph.nrows {..}`.

{: .note-title}
> Common Restrictions on loop ranges
>
> The playground allows you to define loop range bounds based on arbitrarily complex expressions, but it is common for implementations of GraphAlg to be more restrictive. Two types of bounds are supported on all GraphAlg implementations:
> 1. Compile-time constant loop bounds (`int(0):(int(10) + int(100)`).
> 2. Loops over matrix dimensions (`M.nrows`).
>
> Support for e.g. bounds based on function parameters is implementation-dependent.

## Accumulating Results
The `+=` operator is used to combine values of one matrix with a previously defined variable in element-wise fashion.
How entries are combined depends on the semiring.
For `int` and `real` the two elements at the same position are summed, while for `bool` logical OR is used.

{:
    data-ga-func="Accumulate"
    data-ga-render="latex"
    data-ga-arg-0="
2, 2, i64;
0, 0, 1;
0, 1, 2;
1, 0, 3;
1, 1, 4;"
    data-ga-arg-1="
2, 2, i64;
0, 0, 5;
0, 1, 6;
1, 0, 7;
1, 1, 8;"
}
```graphalg
func Accumulate(
        a: Matrix<s, s, int>,
        b: Matrix<s, s, int>) -> Matrix<s, s, int> {
    // EXPERIMENT: Try the equivalent statement:
    // a = a (.+) b;
    a += b;
    return a;
}
```

## Matrix Multiplication
Matrix multiplication is a key building block for many graph algorithms.
As any linear algebra textbook will tell you, matrices can only be multiplied if the number of columns on the left-hand side matches the number of rows of the right-hand side.
The GraphAlg compiler will check this for you automatically, based on dimension symbols in the function parameters.

```graphalg
func InvalidDimensions(
        a: Matrix<s, t, int>,
        b: Matrix<u, t, int>) -> Matrix<s, t, int> {
    return a * b;
}
```

One pattern that is very common, but does not strictly adhere to these restrictions, is *vector-matrix multiplication*.
Consider two variables:
- `a: Vector<s, int>`
- `b: Matrix<s, s, int>`

We have discussed before how `Vector<s, int>` is really just `Matrix<s, 1, int>`, so we have in fact:
- `a: Matrix<s, 1, int>`
- `b: Matrix<s, s, int>`

Then the expression `a * b` is not valid, because `a` has only `1` column while `b` has `s` rows.
For the multiplication to be valid, we must transpose `a` first. If we want the results to have the same shape as `a`, then we must also transpose the result again, and we end up with `(a.T * b).T` (`.T` computes a transpose in GraphAlg).
Because this pattern is very common, GraphAlg allows you to write `a * b`, and taking care of the transpose operations automatically.
You can try this yourself by modifying the example below.

{:
    data-ga-func="VectorMatrixMul"
    data-ga-render="latex"
    data-ga-arg-0="
2, 1, i64;
0, 0, 1;
1, 0, 2;"
    data-ga-arg-1="
2, 2, i64;
0, 0, 3;
0, 1, 4;
1, 0, 5;
1, 1, 6;"
}
```graphalg
func VectorMatrixMul(
        a: Vector<s, int>,
        b: Matrix<s, s, int>) -> Vector<s, int> {
    return (a.T * b).T;
}
```

## Lord of the Semirings
Until now, we have referred to the [*semiring*](https://en.wikipedia.org/wiki/Semiring) of a matrix as the type of the elements.
While the semiring indeed defines the element type, it also defines additional properties of the matrix:
- An *addition operator* $\oplus$, with an identity element called $0$.
- A *multiplication operator* $\otimes$, with an identity element called $1$.

Matrix operations such as the accumulation and matrix multiplication we have seen before use the addition and multiplication operators of the semiring.
Semirings `int` and `real` use the natural definitions of the addition and multiplication operators.
For `bool` the addition and multiplication operators are logical OR and logical AND, respectively.

GraphAlg also includes a more exotic family of semirings called [*tropical semirings*](https://en.wikipedia.org/wiki/Tropical_semiring).
In a tropical semiring, the addition and multiplication operator are defined as:

```math
\begin{align}
    a \oplus b &= \min(a, b) \\
    a \otimes b &= a + b
\end{align}
```

At this point you may wonder, what is all this exotic math good for?
To answer that question, consider the algorithm below.

{:
    data-ga-func="SSSP"
    data-ga-arg-0="
        10, 10, !graphalg.trop_f64;
        0, 1, 0.5;
        0, 2, 5.0;
        0, 3, 5.0;
        1, 4, 0.5;
        2, 3, 2.0;
        4, 5, 0.5;
        5, 2, 0.5;
        5, 9, 23.0;
        6, 0, 1.0;
        6, 7, 3.2;
        7, 9, 0.2;
        8, 9, 0.1;
        9, 6, 8.0;"
    data-ga-arg-1="
        10, 1, !graphalg.trop_f64;
        0, 0, 0;"
    data-ga-result-render="vertex-property"
}
```graphalg
func SSSP(
    graph: Matrix<s, s, trop_real>,
    source: Vector<s, trop_real>) -> Vector<s, trop_real> {
  dist = source;
  for i in graph.nrows {
    dist += dist * graph;
  }
  return dist;
}
```

Careful comparison with the earlier `Reachability` algorithm reveals that the algorithms are structurally identical: Repeated matrix multiplication and accumulation in a loop.
The main difference is the semiring used (`trop_real` instead of `bool`).
`trop_real` is the name for the tropical semiring over real numbers, the tropical equivalent of the `real` semiring.
By using floating-point values rather than booleans, we can record not just that a node is connected, but also keep track of the distance from the source.
The use of $+$ for multiplication means we add the cost of edges the current distance from the source, whereas using $\min$ for addition ensures that we keep only the shortest distance.

TODO: Links to learning more about tropical semirings.

## The Real Deal: PageRank
TODO: Write about PageRank, and where to find docs on all supported operations.

Consult the [list of all operations](../spec/operations) available in GraphAlg.

{:
    data-ga-func="PR"
    data-ga-arg-0="
        50, 50, i1;
        0, 18;
        0, 20;
        0, 21;
        0, 26;
        0, 30;
        0, 36;
        0, 44;
        0, 47;
        1, 2;
        1, 19;
        1, 38;
        1, 45;
        2, 5;
        2, 9;
        2, 31;
        2, 40;
        2, 44;
        3, 14;
        4, 14;
        4, 15;
        4, 17;
        4, 27;
        4, 46;
        5, 48;
        6, 5;
        6, 26;
        6, 42;
        6, 45;
        7, 4;
        7, 20;
        7, 28;
        7, 29;
        7, 31;
        7, 42;
        8, 15;
        8, 17;
        8, 20;
        8, 27;
        8, 29;
        8, 34;
        8, 39;
        9, 8;
        9, 12;
        9, 27;
        9, 28;
        9, 32;
        10, 2;
        10, 38;
        11, 46;
        11, 49;
        12, 6;
        12, 11;
        12, 16;
        12, 31;
        12, 47;
        13, 3;
        13, 19;
        13, 20;
        13, 34;
        13, 37;
        13, 39;
        14, 7;
        14, 23;
        14, 30;
        14, 34;
        14, 43;
        16, 4;
        16, 8;
        16, 10;
        16, 15;
        16, 25;
        16, 36;
        17, 0;
        17, 11;
        17, 27;
        17, 29;
        17, 43;
        17, 44;
        17, 46;
        17, 49;
        18, 9;
        18, 10;
        18, 12;
        18, 26;
        18, 37;
        19, 14;
        19, 24;
        20, 21;
        20, 26;
        20, 30;
        20, 31;
        20, 39;
        21, 18;
        21, 25;
        21, 26;
        21, 30;
        22, 21;
        22, 34;
        22, 35;
        22, 37;
        22, 39;
        22, 45;
        22, 46;
        23, 8;
        23, 12;
        23, 14;
        23, 33;
        23, 35;
        23, 49;
        24, 7;
        24, 23;
        24, 29;
        24, 33;
        24, 40;
        24, 46;
        25, 6;
        25, 30;
        25, 36;
        25, 39;
        25, 43;
        25, 46;
        26, 30;
        26, 32;
        26, 42;
        27, 7;
        27, 31;
        27, 41;
        27, 44;
        28, 0;
        28, 1;
        28, 11;
        28, 13;
        28, 15;
        28, 18;
        28, 19;
        28, 35;
        29, 8;
        29, 23;
        29, 33;
        29, 43;
        30, 10;
        30, 16;
        30, 31;
        30, 38;
        30, 45;
        30, 46;
        31, 1;
        31, 27;
        31, 28;
        31, 29;
        31, 30;
        32, 6;
        32, 7;
        32, 8;
        32, 9;
        32, 31;
        32, 33;
        32, 36;
        33, 25;
        33, 47;
        34, 2;
        34, 9;
        34, 16;
        34, 23;
        34, 25;
        34, 27;
        34, 32;
        34, 40;
        35, 19;
        35, 20;
        35, 28;
        35, 31;
        35, 45;
        36, 0;
        36, 4;
        36, 8;
        36, 12;
        36, 22;
        36, 23;
        37, 1;
        37, 21;
        37, 49;
        38, 5;
        38, 7;
        38, 19;
        38, 27;
        38, 29;
        38, 46;
        38, 47;
        39, 4;
        39, 6;
        39, 7;
        39, 10;
        39, 32;
        39, 33;
        39, 36;
        39, 48;
        40, 23;
        40, 42;
        42, 0;
        42, 1;
        42, 10;
        42, 14;
        42, 16;
        42, 28;
        42, 37;
        42, 46;
        43, 10;
        43, 12;
        43, 14;
        44, 4;
        44, 10;
        44, 11;
        44, 20;
        44, 23;
        45, 22;
        45, 23;
        45, 25;
        45, 30;
        45, 35;
        45, 40;
        46, 7;
        46, 13;
        46, 15;
        46, 27;
        46, 28;
        46, 33;
        46, 34;
        46, 39;
        46, 41;
        46, 45;
        46, 49;
        47, 7;
        47, 18;
        47, 29;
        47, 34;
        47, 37;
        47, 42;
        47, 49;
        48, 6;
        48, 7;
        48, 16;
        48, 17;
        49, 3;
        49, 27;
        49, 46;"
    data-ga-result-render="vertex-property"
}
```graphalg
func withDamping(degree:int, damping:real) -> real {
    return cast<real>(degree) / damping;
}

func PR(graph: Matrix<s1, s1, bool>) -> Vector<s1, real> {
    damping = real(0.85);
    iterations = int(10);
    n = graph.nrows;
    teleport = (real(1.0) - damping) / cast<real>(n);
    rdiff = real(1.0);

    d_out = reduceRows(cast<int>(graph));

    d = apply(withDamping, d_out, damping);

    connected = reduceRows(graph);
    sinks = Vector<bool>(n);
    sinks<!connected>[:] = bool(true);

    pr = Vector<real>(n);
    pr[:] = real(1.0) / cast<real>(n);

    for i in int(0):iterations {
        sink_pr = Vector<real>(n);
        sink_pr<sinks> = pr;
        redist = (damping / cast<real>(n)) * reduce(sink_pr);

        w = pr (./) d;

        pr[:] = teleport + redist;
        pr += cast<real>(graph).T * w;
    }

    return pr;
}
```

<script src="/playground/editor.bundle.js"></script>

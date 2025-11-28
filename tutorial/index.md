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
    a += b;
    return a;
}
```

## Matrix Multiplication

<script src="/playground/editor.bundle.js"></script>

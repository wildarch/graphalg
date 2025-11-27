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
> If you write an invalid program, the editor will underline the part of your
> program that is incorrect. Hover over the underlined code to see the error
> message, or click *Run* to print error messages to the output.

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
A function can have any number of parameter, but it must have **exactly one return value**.
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

## Bringing Linear Algebra Into the Mix
Our example programs so far have only used the `int` type so far.
There are also `bool` (`false` or `true`) and `real` (floating-point numbers, 64-bit).
To capture the connected nature of graphs, however, we need more than simple scalar types.
GraphAlg represents graphs as adjacency matrices.

## TODO
```graphalg
func Reachability(
        graph: Matrix<s, s, bool>,
        source: Vector<s, bool>) -> Matrix<s, s, bool> {
    v = source;
    for i in graph.nrows {
        v += v * graph;
    }

    return v;
}
```

This algorithm implements a [reachability analysis](https://en.wikipedia.org/wiki/Reachability).
It determines which nodes in the graph are connected through any number of edges (reachable) from any of the *source* vertices.

GraphAlg programs are a collection of functions.
They are defined using the `func` keyword.
GraphAlg is *statically typed*: you must assign types to the parameters of a function, and to the return type.
`graph: Matrix<s, s, bool>` on line 2 defines a parameter named `graph` with type `Matrix<s, s, int>`.



<script src="/playground/editor.bundle.js"></script>

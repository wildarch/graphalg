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

GraphAlg programs are a collection of functions.
They are defined using the `func` keyword.
GraphAlg is *statically typed*: you must assign types to the parameters of a function, and to the return type.
`a: int` on line 1 defines a parameter named `a` with type `int`.
The return type of this function is also `int`, as indicated by `-> int` following the parameters.

After the return type and inside of the `{ .. }` is the implementation of the function.
In this case the implementation is rather trivial: Take the value of `a` and add one to it.

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

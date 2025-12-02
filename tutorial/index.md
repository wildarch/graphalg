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

{: .note-title }
> Running example programs
>
> The examples shown in this tutorial run inside your own browser using
> [WebAssembly](https://webassembly.org/).
> You can modify and run examples as much as you want.
>
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

The `AddOne` function uses the `int` type for both the parameter and the return type.
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
In graph algorithms it is very common to bound loops based on the number of nodes in the graph, so GraphAlg provides shorthand syntax `for i in graph.nrows {..}`.
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

If you are interested to learn more about the use of semirings (and linear algebra more broadly) in the context of graph algorithms, we recommend the book [*Graph Algorithms in the Language of Linear Algebra*](https://epubs.siam.org/doi/book/10.1137/1.9780898719918) by Jeremy Kepner and John Gilbert.
We can also recommend [various resources](https://github.com/GraphBLAS/GraphBLAS-Pointers) related to [GraphBLAS](https://graphblas.org/), an API that provides building blocks for graph algorithms also based on linear algebra.
The more conceptual of those resources are also applicable to GraphAlg.

{: .note-title }
> GraphBLAS vs. GraphAlg
>
> GraphBLAS defines a C library with sparse linear algebra routines, so it operates at a lower abstraction level than GraphAlg.
> Operations are similar, but there are subtle and important differences between the two.
> Work is in progress to build a GraphBLAS target for GraphAlg, which would allow running GraphAlg programs using a runtime based on GraphBLAS.
> If you want to follow progress on the GraphBLAS integration, you can follow the [GraphAlg Repository on GitHub](https://github.com/wildarch/graphalg).

## The Real Deal: PageRank
Let us now move to a more complex algorithm.
[PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) is a well-known and widely used algorithm for computing the *importance* of nodes in a graph.
It was made famous by Google, who used it to the measure the importance of websites in the graph that is the World Wide Web.
The algorithm is still in wide use today, for example to [analyze the influence of scientific publications](https://graph.openaire.eu/docs/graph-production-workflow/indicators-ingestion/impact-indicators/#pagerank-pr--influence).
Below you can find an implementation of PageRank in GraphAlg.

{:
    data-ga-func="PR"
    data-ga-arg-0="
        11, 11, i1;
        1, 2;
        2, 1;
        3, 0;
        3, 1;
        4, 1;
        4, 3;
        5, 1;
        5, 4;
        6, 1;
        6, 4;
        7, 1;
        7, 4;
        8, 1;
        8, 4;
        9, 4;
        10, 4;"
    data-ga-result-render="vertex-property"
}
```graphalg
func withDamping(degree:int, damping:real) -> real {
    return cast<real>(degree) / damping;
}

func PR(graph: Matrix<s, s, bool>) -> Vector<s, real> {
    // A commonly-used value for the damping factor (85%).
    damping = real(0.85);
    // Run for 10 iterations
    iterations = int(10);

    // Number of nodes in the graph
    n = graph.nrows;

    // Per-node probability that a random surfer will jump to that
    // node.
    teleport = (real(1.0) - damping) / cast<real>(n);

    // Per-node out degree (number of outgoing edges)
    d_out = reduceRows(cast<int>(graph));
    // .. with damping applied.
    d = apply(withDamping, d_out, damping);

    // Sinks are nodes that have no outgoing edges.
    // Sometimes also called 'dangling nodes' or 'dead-ends'.
    connected = reduceRows(graph);
    sinks = Vector<bool>(n);
    sinks<!connected>[:] = bool(true);

    // Initial PageRank score: equally distributed over all nodes.
    pr = Vector<real>(n);
    pr[:] = real(1.0) / cast<real>(n);

    for i in int(0):iterations {
        // total PageRank score across all sinks.
        sink_pr = Vector<real>(n);
        sink_pr<sinks> = pr;

        // redistribute PageRank score from sinks
        redist = (damping / cast<real>(n)) * reduce(sink_pr);

        // Previous PageRank score divided by the (damped) out degree.
        // This gives us a per-node score amount to distributed to its
        // outgoing edges.
        w = pr (./) d;

        // Initialize next PageRank scores with the uniform teleport
        // probability and the amount redistributed from the sinks.
        pr[:] = teleport + redist;

        // Distribute the previous PageRank scores over the outgoing
        // edges (and add to the new PageRank score).
        pr += cast<real>(graph).T * w;
    }

    return pr;
}
```

If you run the algorithm, you will see that node 1 is the most influentation node in the graph (it has the highest rank).
This makes sense given that many nodes have an edge to node 1.
Notice, though, that node 2 is also highly influential, yet it has very few incoming edges.
Its high ranking comes from the influential node 1, whose only outgoing edge is to node 2, boosting the influence of node 2.

The PageRank implementation presented above uses a few new language constructs:
- `cast<T>` (line 2, 9, etc.) casts the input to a different semiring `T`.
  For example, `cast<real>` on line 2 promotes an integer value to floating-point.
- `reduceRows(M)` (line 12) collapses a matrix `M` into a column vector, summing elements using the addition operator.
- `apply` (line 14) applies a scalar function to every element of a matrix.
  An additional second scalar argument can be specified that is passed as the second argument to the function.
- `M[:] = c` (line 18, 21, 30) replaced every element of `M` by scalar value `c`.
- `A<M> = B` (line 18, 25) assigns elements from `B` to the same position in `A` iff `M` has a nonzero value at that position.
- `reduce` (line 25) sums all elements of a matrix to scalar.
- `A (./) B` (line 28) represents elementwise division.
  It applies the division operator to each position of `A` and `B`.
  The output is defined as $O_{ij} = A_{ij} / B_{ij}$.
  Other operators such as `+` and `*` and even arbitary functions (`A (.myFunc) B`) also support elementwise application.

For a detailed explanation of these and other GraphAlg operations, see the [operations](../spec/operations) section of the language specification.

## Where Next?
This concludes our introduction to the GraphAlg language.
Where you go from here depends on your needs:
- Do you want to experiment more with different algorithms?
  Check out [GraphAlg Playground](../playground).
- Are you ready to move beyond experiments and use GraphAlg to analyze large graphs?
  See the [available implementations](../integration/available).
- If you want to learn more about the design, features and theoretical foundations for the GraphAlg language, you can find a more details in the [language specification](../spec).
  In particular, you can find a full overview of all [operations](../spec/operations) available in the GraphAlg there.
- Are you a system developer looking to integrate GraphAlg? See our guide for [new integrators](../integration/new_integration).

<script src="{{ '/playground/editor.bundle.js' | relative_url }}"></script>

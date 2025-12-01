---
title: Developing a New Integration
layout: page
parent: Integrating
nav_order: 2
---

# Integrating GraphAlg into Existing Systems
GraphAlg is specifically designed to be integrated into existing systems.
The GraphAlg [compiler](https://github.com/wildarch/graphalg/tree/main/compiler) source code is freely available under a permissive license.
The compiler can be integrated into other systems as a C++ library.
Below we describe the three main integration points you can use depending on the properties of the target system.

## Integrating The Reference Backend
To minimize the required porting effort you can use the full compiler including the reference backend.
An example of this approach is the [GraphAlg Playground](https://github.com/wildarch/graphalg/tree/main/playground) (see the [C++ component](https://github.com/wildarch/graphalg/tree/main/playground/cpp) in particular).
An important caveat is that the reference is only designed to handle very small example-sized graphs.
It will be slow and use a lot of memory if you try to use it with a larger graph (>100 nodes or edges).

## Integrating From Relational Algebra
If your system uses a relational algebra representation internally, or something akin to it, you can leverage the [conversion to relational algebra](../spec/core/relalg).
If your system supports common arithmetic operations and aggregator functions, you only need to implement suitable a loop operator.
All other GraphAlg operations can be converted into standard relational algebra operations.

{: .note-title }
> Example integration
>
> This approach is used by [AvantGraph](https://avantgraph.io/), but source code for this integration is not publicly available (yet).
> When this code becomes freely available, or if another integration using the same approach becomes available, we will link to that here.

## Integrating From GraphAlg Core
The [Core language](../spec/core) has only a small number of high-level [operations](../spec/core/operations) that need to be implemented.
This may be a more suitable integration point if your system does not target relational algebra.
To do this you should use the provided [parser]() and run the [pipeline](https://github.com/wildarch/graphalg/blob/main/compiler/src/graphalg/GraphAlgToCorePipeline.cpp) to lower to GraphAlg Core.

{: .note-title }
> Example integration
>
> We do not currently have an example integration for this approach, although a GraphBLAS backend is planned that would use this strategy.

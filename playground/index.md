---
title: GraphAlg Playground
layout: page
nav_order: 3
---

# GraphAlg Playground
Compile and execute GraphAlg programs in your browser!

{:
    data-ga-func="MatMul"
    data-ga-arg-0="
        2, 2, i64;
        0, 0, 3;
        0, 1, 5;
        1, 0, 7;
        1, 1, 11;"
    data-ga-arg-1="
        2, 2, i64;
        0, 0, 13;
        0, 1, 17;
        1, 0, 19;
        1, 1, 23;"
}
```graphalg
func MatMul(
        lhs: Matrix<s,s,int>,
        rhs: Matrix<s,s,int>) -> Matrix<s,s,int> {
    return lhs * rhs;
}
```

## Breadth-First Search

{:
    data-ga-func="BFS"
    data-ga-arg-0="
        10, 10, i1;
        0, 1;
        0, 2;
        1, 2;
        1, 3;
        1, 4;
        2, 0;
        3, 5;
        3, 6;
        3, 7;
        4, 0;
        4, 1;
        5, 3;
        5, 7;
        7, 0;
        7, 1;
        7, 2;
        8, 9;"
    data-ga-arg-1="
        10, 1, i1;
        0, 0;"
}
```graphalg
func setDepth(b:bool, iter:int) -> int {
    return cast<int>(b) * (iter + int(2));
}

func BFS(
        graph: Matrix<s, s, bool>,
        source: Vector<s, bool>) -> Vector<s, int> {
    v = Vector<int>(graph.nrows);
    v<source>[:] = int(1);

    frontier = source;
    reach = source;

    for i in graph.nrows {
        step = Vector<bool>(graph.nrows);
        step<!reach> = frontier * graph;

        v += apply(setDepth, step, i);

        frontier = step;
        reach += step;
    } until frontier.nvals == int(0);

    return v;
}
```

<script src="editor.bundle.js"></script>

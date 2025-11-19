---
title: GraphAlg Playground
layout: page
nav_order: 3
---

# GraphAlg Playground
Compile and execute GraphAlg programs in your browser!

{: .warning-title }
> Early Preview
>
> The examples below are highly experimental and may malfunction.

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

## Label Propagation

{:
    data-ga-func="CDLP"
    data-ga-arg-0="
        8, 8, i1;
        0, 1;
        0, 2;
        0, 6;
        1, 0;
        1, 2;
        2, 0;
        2, 1;
        3, 4;
        3, 5;
        4, 3;
        4, 5;
        4, 6;
        5, 4;
        5, 6;
        6, 4;
        6, 5;
        6, 7;
        7, 5;"
}
```graphalg
func isMax(v: int, max: trop_max_int) -> bool {
  return (cast<trop_max_int>(v) == max)
      * (v != zero(int));
}

func CDLP(graph: Matrix<s, s, bool>) -> Matrix<s, s, bool> {
    iterations = int(5);
    id = Vector<bool>(graph.nrows);
    id[:] = bool(true);
    L = diag(id);

    for i in int(0):iterations {
        step_forward = cast<int>(graph) * cast<int>(L);
        step_backward = cast<int>(graph.T) * cast<int>(L);
        step = step_forward (.+) step_backward;

        // Max per row
        max = reduceRows(cast<trop_max_int>(step));

        // Broadcast to all columns
        b = Vector<trop_max_int>(graph.ncols);
        b[:] = one(trop_max_int);
        max_broadcast = max * b.T;

        // Matrix with true at every position where L has max element.
        step_max = step (.isMax) max_broadcast;

        // Keep only one assigned label per vertex.
        // The implementation always picks the one with the lowest id.
        L = pickAny(step_max);
    }

    // Map isolated nodes to their own label.
    connected = reduceRows(graph) (.+) reduceRows(graph.T);
    isolated = Vector<bool>(graph.nrows);
    isolated<!connected>[:] = bool(true);
    L = diag(isolated) (.+) L;

    return L;
}
```

## PageRank

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

## Single-Source Shortest Paths

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
        10, 1, i1;
        0, 0;"
}
```graphalg
func SSSP(
    graph: Matrix<s1, s1, trop_real>,
    source: Vector<s1, bool>) -> Vector<s1, trop_real> {
  v = cast<trop_real>(source);
  for i in graph.nrows {
    v += v * graph;
  }
  return v;
}
```

## Weakly Connected Components

{:
    data-ga-func="WCC"
    data-ga-arg-0="
        9, 9, i1;
        0, 1;
        0, 2;
        1, 0;
        1, 2;
        1, 3;
        3, 1;
        5, 6;
        5, 7;
        6, 5;
        8, 2;"
}
```graphalg
func WCC(graph: Matrix<s, s, bool>) -> Matrix<s, s, bool> {
  id = Vector<bool>(graph.nrows);
  id[:] = bool(true);
  label = diag(id);

  for i in graph.nrows {
    // Keep current label
    alternatives = label;
    // Labels reachable with a forward step
    alternatives += graph * label;
    // Labels reachable with a backward step
    alternatives += graph.T * label;

    // Select a new label
    label = pickAny(alternatives);
  }

  return label;
}
```

<script src="editor.bundle.js"></script>

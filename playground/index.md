---
title: GraphAlg Playground
layout: page
nav_order: 4
---

# GraphAlg Playground
Compile and execute GraphAlg programs in your browser!

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
    data-ga-editor="playground"
    data-ga-result-render="vertex-property"
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

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/default.min.css">
<script src="editor.bundle.js"></script>

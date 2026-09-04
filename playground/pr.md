---
title: GraphAlg Playground (PageRank example)
layout: page
parent: GraphAlg Playground
nav_order: 1
---

# GraphAlg Playground
Compile and execute GraphAlg programs in your browser!

{:
    data-ga-func="PR"
    data-ga-arg-0="11, 11, i1; 1, 2; 2, 1; 3, 0; 3, 1; 4, 1; 4, 3; 5, 1; 5, 4; 6, 1; 6, 4; 7, 1; 7, 4; 8, 1; 8, 4; 9, 4; 10, 4;"
    data-ga-editor="playground"
    data-ga-result-render="vertex-property"
}
```graphalg
// With redist from sinks
func withDamping(degree:int, damping:real) -> real {
  return cast<real>(degree) / damping;
}

func PR(graph: Matrix<s, s, bool>) -> Vector<s, real> {
  damping = real(0.85);
  iterations = int(10);
  n = graph.nrows;
  teleport = (real(1.0) - damping) / cast<real>(n);

  d_out = reduceRows(cast<int>(graph));
  d = apply(withDamping, d_out, damping);

  // NEW: Find sinks
  connected = reduceRows(graph);
  sinks = Vector<bool>(n);
  sinks<!connected>[:] = bool(true);

  pr = Vector<real>(n);
  pr[:] = real(1.0) / cast<real>(n);

  for i in int(0):iterations {
    // NEW: compute redist amount per vertex.
    sink_pr = Vector<real>(n);
    sink_pr<sinks> = pr;
    redist = (damping / cast<real>(n)) * reduce(sink_pr);

    w = pr (./) d;

    // NEW: Add redist value.
    pr[:] = teleport + redist;
    pr += cast<real>(graph).T * w;
  }

  return pr;
}
```

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/default.min.css">
<script src="editor.bundle.js"></script>

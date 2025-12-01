---
title: "The GraphAlg language"
layout: home
nav_order: 1
---

# The GraphAlg Language
GraphAlg is a language for graph algorithms designed to be embedded into databases.

{:
    data-ga-func="PageRank"
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

func PageRank(graph: Matrix<s, s, bool>) -> Vector<s, real> {
    damping = real(0.85);
    iterations = int(10);
    n = graph.nrows;
    teleport = real(0.15) / cast<real>(n);

    d_out = reduceRows(cast<int>(graph));
    d = apply(withDamping, d_out, damping);

    pr = Vector<real>(n);
    pr[:] = real(1.0) / cast<real>(n);

    for i in int(0):iterations {
        w = pr (./) d;
        pr[:] = teleport;
        pr += cast<real>(graph).T * w;
    }

    return pr;
}
```

Are you new to GraphAlg?
Write your first GraphAlg program and learn more about the language using the interactive [tutorial](./tutorial).

You can experiment with GraphAlg in our [Playground](./playground), or use a [system with GraphAlg support](./integration/available).

For a detailed overview of the GraphAlg language, see the [language specification](./spec).

<script src="{{ '/playground/editor.bundle.js' | relative_url }}"></script>

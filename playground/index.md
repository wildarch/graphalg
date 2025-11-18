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

<script src="editor.bundle.js"></script>

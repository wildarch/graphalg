---
title: Relation to for-MATLANG
layout: page
parent: Core Language
nav_order: 4
---


# Relation to for-MATLANG
GraphAlg Core is closely related to for-MATLANG (see the paper *[Expressive power of linear algebra query languages by Geerts et al.](https://arxiv.org/abs/2010.13717)*).
The main departure from that paper is the definition of the loop construct:
* for-MATLANG only allows a single state variable in the loop, while GraphAlg allows multiple.
* for-MATLANG exposes *canonical vectors* inside the loop. GraphAlg instead provides a `pickAny` operation for leader election.

When constrained to a single state variable, the GraphAlg has the same expressive power as for-MATLANG.
We show this by simulating canonical vectors in GraphAlg, and then by simulating `pickAny` in for-MATLANG.

Canonical vectors can be simulated in GraphAlg using `pickAny`:
```
func canonicalVectors(X: Matrix<s, s, bool>) -> Matrix<s, s, bool> {
    // State for the canonical vector
    canon = Vector<bool>(graph.nrows);
    canon[:] = bool(true);

    for i in v.nrows {
        canonical_vector = pickAny(canon.T).T;

        // BEGIN - Inner body of the for-MATLANG loop
        X = ...;
        // END - Inner body of the for-MATLANG loop

        v<canonical_vector>[:] = bool(false);
    }

    return X;
}
```

Conversely, `pickAny` can be simulated in for-MATLANG using canonical vectors:
```
// Implements out = pickAny(A);
out = zero_matrix_of_type(A)
for v_i: // row per row
    for w_j:
        B = diag(v_i) * A * diag(w_j) // B_ij = A_ij and zero elsewhere
        s = v_i^T * A * w_j // scalar containing B_ij = A_ij
        if s != 0:
            break
    out += B // row i is done
```

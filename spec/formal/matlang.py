#!/usr/bin/env python3
import numpy as np

def transp(M):
    return M.T

def one(M):
    n = np.size(M, axis=0)
    return np.ones((n, 1), M.dtype)

def diag(M):
    n = np.size(M, axis=0)
    assert (M.shape == (n, 1))
    return np.diag(M[:, 0])

def matmul(A, B):
    return np.matmul(A, B)

# apply: vectorize

def ffor(v, M, f, init=False):
    if init:
        A = M
    else:
        A = np.zeros(M.shape, dtype=M.dtype)

    n = np.size(v, axis=0)
    assert (v.shape == (n, 1))
    for i in range(n):
        v = np.zeros((n, 1), dtype=M.dtype)
        v[i, 0] = 1
        A = f(v, A)

    assert (A.shape == M.shape)
    return A

def gfor(f, Md, *init):
    # List of matrices
    state = init

    n = np.size(Md, axis=0)
    for i in range(n):
        v = np.zeros((n, 1), dtype=Md.dtype)
        v[i, 0] = 1
        state = f(v, *state)

    return state[0]

def emax(M):
    return ffor(one(M), one(M), lambda v, X: v)

def S_lte(M):
    def S_lte_inner(v, X):
        X_emax = matmul(X, emax(M))
        X_emax_v = X_emax + v
        X_emax_v_v = matmul(X_emax_v, transp(v))
        v_emax = matmul(v, transp(emax(M)))
        return X + X_emax_v_v + v_emax

    # Output is square
    out_shape = diag(one(M))
    X = ffor(one(M), out_shape, S_lte_inner)
    # FIX not in original paper
    res = X - one(X) * transp(emax(X))
    return res

def ident(M):
    return diag(one(M))

def S_lt(M):
    return S_lte(M) - ident(M)

def pickAny(A):
    def pick(y, d, p):
        if d == 0:
            return p
        else: return y
    pickv = np.vectorize(pick)
    v = one(A)
    w = one(transp(A))
    X = A
    Y = transp(w)
    B = matmul(w, transp(w)) 
    def pickAny_row(v, X):
        # R = V.T * A
        R = matmul(transp(v), A)
        def pickAny_col(w, Y):
            # D = Y * B
            D = matmul(Y, B)
            # P = R * diag(w)
            P = matmul(R, diag(w))
            assert(Y.shape == D.shape)
            assert(Y.shape == P.shape)
            # apply[pick](Y, D, P)
            return pickv(Y, D, P)
        # for w,Y. ...
        row = ffor(w, Y, pickAny_col)
        # X + v * (for w,Y. ...)
        return X + matmul(v, row)
    # for v,X. ...
    return ffor(v, X, pickAny_row)

def emin(M):
    # NOTE: Different definition than the one in the for-ML paper
    return transp(pickAny(transp(one(M))))

def shift(M):
    return pickAny(S_lt(M))

def rotate(M):
    return shift(M) + matmul(emax(M), transp(emin(M)))

def max_element_vec(M):
    R = rotate(M)
    def max_element_inner(v, X):
        # apply[max](M, R*X)
        return np.maximum(M, matmul(R, X))
    X = ffor(one(M), M, max_element_inner)
    return matmul(transp(emax(M)), X)

def max_per_row(M):
    def max_row(v, X):
        # One row of M
        B = matmul(transp(v), M)
        B_max = max_element_vec(transp(B))
        B_max = matmul(v, B_max)
        return X + B_max
    return ffor(one(M), one(M), max_row)

# M = np.array([
#     [1, 2],
#     [4, 3],
#     [5, 6],
# ])
# print(max_per_row(M))

def floyd_warshall(D):
    def fw_inner(v, D):
        to_k = matmul(D, v)
        fo_k = matmul(transp(v), D)
        # Workaround because we don't have min.+ semiring
        via_k = matmul(to_k, transp(one(v))) + matmul(one(v), fo_k)

        # apply[min](D, via_k)
        return np.minimum(D, via_k)
    return ffor(one(D), D, fw_inner, init=True)


def floyd_warshall_path(L, D):
    # TODO: init
    P = np.array([
        [1, 0, 1, 0],
        [2, 2, 2, 0],
        [0, 0, 3, 3],
        [0, 4, 0, 4],
    ])

    def fw_inner(v, P, D):
        to_k = matmul(D, v)
        fo_k = matmul(transp(v), D)
        via_k = matmul(to_k, transp(one(v))) + matmul(one(v), fo_k)

        newD = np.minimum(D, via_k)

        # Mod to get paths
        updated = newD != D
        newP = matmul(one(v), matmul(transp(v), P))
        P = np.where(newD != D, newP, P)
        return P, newD
    return gfor(fw_inner, P, P, D)

# inf = 1e32
# D = np.array([
#     [0.0, inf, -2.0, inf],
#     [4, 0, 3, inf],
#     [inf, inf, 0.0, 2.0],
#     [inf, -1, inf, 0],
# ])
# L = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
# ])
# print(floyd_warshall_path(L, D))
# print(floyd_warshall(D))

def argmin(M):
    R = rotate(M)
    def min_element_inner(v, X):
        # apply[min](M, R*X)
        return np.minimum(M, matmul(R, X))
    X = ffor(one(M), M, min_element_inner, init=True)
    return X == M

# INF = 1e32
# def prims(A, s):
#     # d = A(s, :)
#     # A is symmetric, no need to transpose
#     d = matmul(A, s)
#     def prims_inner(v, d, s):
#         u = argmin((s * INF) + d)
#         # TODO: simulate if
#         if np.all(u == one(u)):
#             return d, s
#         # Keep only one
#         u = transp(pickAny(transp(u)))
#         s = s + u
#         d = np.minimum(d, matmul(A, u))
#         return d, s
#     return gfor(prims_inner, A, d, s)

# A = np.array([
#     [INF, 2, INF, 1],
#     [2, INF, INF, 2],
#     [INF, INF, INF, 3],
#     [1, 2, 3, INF],
# ])

# s = np.array([
#     [1],
#     [0],
#     [0],
#     [0],
# ])

# print(prims(A, s))

def matmul_simulate(A, B):
    shape = matmul(A, B)
    def per_row(v, X):
        A_row = matmul(transp(v), A)
        def per_col(w, Y):
            B_col = matmul(B, w)
            # element-wise multiply row from A with column from B
            mul = np.multiply(transp(A_row), B_col)
            # Sum to a single cell value
            cell = matmul(transp(one(mul)), mul)
            # Map to the expected output position
            mat = matmul(v, matmul(cell, transp(w)))
            # NOTE: transpose mat here because we are iterating over result in 
            # transposed form (to visit columns of B rather than rows)
            return Y + transp(mat)
        # TODO: fix v
        return X + transp(ffor(one(transp(shape)), transp(shape), per_col))
    return ffor(one(shape), shape, per_row)

def matmul_minplus(A, B):
    shape = matmul(A, B)
    def per_row(v, X):
        A_row = matmul(transp(v), A)
        def per_col(w, Y):
            B_col = matmul(B, w)
            # element-wise add row from A with column from B
            add = transp(A_row) + B_col
            # Get minimum value
            cell = min_element_vec(add)
            # Map to the expected output position
            mat = matmul(v, matmul(cell, transp(w)))
            # NOTE: transpose mat here because we are iterating over result in 
            # transposed form (to visit columns of B rather than rows)
            return Y + transp(mat)
        return X + transp(ffor(one(transp(shape)), transp(shape), per_col))
    return ffor(one(shape), shape, per_row)

def min_element_vec(M):
    R = rotate(M)
    def min_element_inner(v, X):
        # apply[min](M, R*X)
        return np.minimum(M, matmul(R, X))
    X = ffor(one(M), M, min_element_inner, init=True)
    return matmul(transp(emax(M)), X)

# A = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
# ])

# B = np.array([
#     [7, 8, 9, 10],
#     [11, 12, 13, 14],
#     [15, 16, 17, 18],
# ])

# print(matmul_minplus(A, B))

M = np.array([
    [1, 2, 3],
    [0, 5, 6],
])

print(pickAny(M))
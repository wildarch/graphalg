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

def ffor(M, f, init=False):
    if init:
        A = M
    else:
        A = np.zeros(M.shape, dtype=M.dtype)

    n = np.size(M, axis=0)
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
    return ffor(one(M), lambda v, X: v)

def S_lte(M):
    def S_lte_inner(v, X):
        X_emax = matmul(X, emax(M))
        X_emax_v = X_emax + v
        X_emax_v_v = matmul(X_emax_v, transp(v))
        v_emax = matmul(v, transp(emax(M)))
        return X + X_emax_v_v + v_emax

    # Output is square
    out_shape = diag(one(M))
    X = ffor(out_shape, S_lte_inner)
    # FIX not in original paper
    res = X - one(X) * transp(emax(X))
    return res

def ident(M):
    return diag(one(M))

def S_lt(M):
    return S_lte(M) - ident(M)

def pickAny(M):
    def pickAny_row(v, X):
        B = matmul(transp(v), M)
        def pick(y, d, p):
            if d == 0:
                return p
            else: return y
        pickv = np.vectorize(pick)
        def pickAny_col(w, Y):
            # D = 1(Y) * 1(Y).T * Y
            D = matmul(one(Y), matmul(transp(one(Y)), Y))
            # P = diag(w) * B.T
            P = matmul(diag(w), transp(B))
            assert(Y.shape == D.shape)
            assert(Y.shape == P.shape)
            # apply[pick](Y, D, P)
            return pickv(Y, D, P)
        row = ffor(transp(B), pickAny_col)
        return X + matmul(v, transp(row))
    return ffor(M, pickAny_row)

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
    X = ffor(M, max_element_inner)
    return matmul(transp(emax(M)), X)

def max_per_row(M):
    def max_row(v, X):
        # One row of M
        B = matmul(transp(v), M)
        B_max = max_element_vec(transp(B))
        B_max = matmul(v, B_max)
        return X + B_max
    return ffor(one(M), max_row)

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
    return ffor(D, fw_inner, init=True)


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

inf = 1e32
D = np.array([
    [0.0, inf, -2.0, inf],
    [4, 0, 3, inf],
    [inf, inf, 0.0, 2.0],
    [inf, -1, inf, 0],
])

L = np.array([
    [1],
    [2],
    [3],
    [4],
])

#print(floyd_warshall_path(L, D))
print(floyd_warshall(D))

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

def ffor(M, f, debug=False):
    A = np.zeros(M.shape, dtype=M.dtype)
    n = np.size(M, axis=0)
    for i in range(n):
        if debug:
            print(f"A{i}")
            print(A)
        v = np.zeros((n, 1), dtype=M.dtype)
        v[i, 0] = 1

        A = f(v, A)
    
    if debug:
        print(f"A{n}")
        print(A)
    
    assert (A.shape == M.shape)
    return A

def emax(M):
    return ffor(M, lambda v, X: v)

def S_lt(M):
    def S_lt_inner(v, X):
        X_emax = matmul(X, emax(M))
        X_emax_v = X_emax + v
        X_emax_v_v = matmul(X_emax_v, transp(v))
        v_emax = matmul(v, transp(emax(M)))
        return X + X_emax_v_v + v_emax

    X = ffor(M, S_lt_inner)
    # FIX not in original paper
    res = X - one(X) * transp(emax(X))
    return res

def pickAny(M):
    def pickAny_row(v, X):
        B = matmul(transp(v), M)
        def pick(y, d, p):
            if d == 0:
                return p
            else: return y
        pickv = np.vectorize(pick)
        def pickAny_col(w, Y):
            D = matmul(one(Y), matmul(transp(one(Y)), Y))
            P = matmul(diag(w), transp(B))
            assert(Y.shape == D.shape)
            assert(Y.shape == P.shape)
            res = pickv(Y, D, P)
            print("Y")
            print(Y)
            print("D")
            print(D)
            print("P")
            print(P)
            return res
        row = ffor(transp(B), pickAny_col)
        print("==== ROW ====")
        return X + matmul(v, transp(row))
    return ffor(M, pickAny_row)

A = np.array([
    [0, 2, 3],
    [4, 5, 6],
    [0, 0, 9],
    [0, 0, 0],
])

print(pickAny(A))
# Y = np.array([
#     [0],
#     [2],
#     [0],
# ])
#print(matmul(one(Y), matmul(transp(one(Y)), Y)))

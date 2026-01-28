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
    return A

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

def emax(M):
    return ffor(M, lambda v, X: v)

def S_lt(M):
    def S_lt_inner(v, X):
        X_emax = matmul(X, emax(M))
        X_emax_v = X_emax + v
        X_emax_v_v = matmul(X_emax_v, transp(v))
        v_emax = matmul(v, transp(emax(M)))
        print(f"X_emax_v_v=\n{X_emax_v_v}")
        print(f"v=\n{v}")
        print(f"emax=\n{transp(emax(M))}")
        print(f"v_emax=\n{v_emax}")
        return X + X_emax_v_v + v_emax

    X = ffor(M, S_lt_inner, debug=True)
    # FIX not in original paper
    res = X - one(X) * transp(emax(X))
    return res

def max(u):
    return transp(u) * emax(u)

u = np.array([
    [1],
    [2],
    [3],
])

def prev(M):
    def prev_inner(v, X):
        part1 = (1 - max(v)) * v * transp(emax(M))
        part2 = X * emax(M) * transp(emax(M))
        part3 = X * emax(M) * transp(v)
        return X + (part1 - part2 + part3)
    return ffor(M, prev_inner)

print(S_lt(np.zeros((10, 10))))
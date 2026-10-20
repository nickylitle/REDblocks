"""Small dense solves for the rigid-joint systems (a handful to a few dozen rows), written out so a joint solve
neither allocates nor calls LAPACK each time step."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _gauss(K, x, n):
    """Solve K x = rhs in place (x holds rhs on entry): Gaussian elimination with partial pivoting, small dense n.
    Returns False if K is singular."""
    for c in range(n):
        p = c; mv = abs(K[c, c])
        for i in range(c + 1, n):
            if abs(K[i, c]) > mv:
                mv = abs(K[i, c]); p = i
        if mv == 0.0 or not np.isfinite(mv):
            return False
        if p != c:
            for k in range(c, n):
                t = K[c, k]; K[c, k] = K[p, k]; K[p, k] = t
            t = x[c]; x[c] = x[p]; x[p] = t
        inv = 1.0 / K[c, c]
        for i in range(c + 1, n):
            f = K[i, c] * inv
            if f != 0.0:
                for k in range(c + 1, n):
                    K[i, k] -= f * K[c, k]
                x[i] -= f * x[c]
    for c in range(n - 1, -1, -1):
        s_ = x[c]
        for k in range(c + 1, n):
            s_ -= K[c, k] * x[k]
        x[c] = s_ / K[c, c]
    return True


@njit(cache=True, nogil=True)
def _lu(K, piv, n):
    """LU factors of K in place (partial pivoting; piv records the row swaps). False if singular."""
    for c in range(n):
        p = c; mv = abs(K[c, c])
        for i in range(c + 1, n):
            if abs(K[i, c]) > mv:
                mv = abs(K[i, c]); p = i
        piv[c] = p
        if mv == 0.0 or not np.isfinite(mv):
            return False
        if p != c:
            for k in range(n):
                t = K[c, k]; K[c, k] = K[p, k]; K[p, k] = t
        inv = 1.0 / K[c, c]
        for i in range(c + 1, n):
            K[i, c] *= inv
            f = K[i, c]
            if f != 0.0:
                for k in range(c + 1, n):
                    K[i, k] -= f * K[c, k]
    return True


@njit(cache=True, nogil=True)
def _lu_solve(K, piv, x, n):
    for c in range(n):
        p = piv[c]
        if p != c:
            t = x[c]; x[c] = x[p]; x[p] = t
    for c in range(n):
        for k in range(c):
            x[c] -= K[c, k] * x[k]
    for c in range(n - 1, -1, -1):
        s_ = x[c]
        for k in range(c + 1, n):
            s_ -= K[c, k] * x[k]
        x[c] = s_ / K[c, c]

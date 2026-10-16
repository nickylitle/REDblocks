"""numba wrapper. Kernels compile with numba when present and run as plain Python otherwise."""
try:
    from numba import njit as _njit
    HAVE_NUMBA = True

    def kernel(f):
        return _njit(cache=True)(f)
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def kernel(f):
        return f

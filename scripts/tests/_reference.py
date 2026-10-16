"""Independent reference solvers used ONLY by the tests. They share no code with simblocks,
so a block passing its test means two different methods agree."""
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import eigh


def fe_beam(L, EI, n_el, bcs, loads, mass_per_L=None):
    """Hermite Euler-Bernoulli beam FE. bcs: dict node -> list of fixed dofs (0 = w, 1 = theta).
    loads: list of (node, dof, value). Returns w, theta at nodes, element end moments, and (optional) eigenfreqs."""
    ne = n_el; nn = ne + 1; le = L / ne
    K = np.zeros((2 * nn, 2 * nn)); Mm = np.zeros_like(K)
    k = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                 [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    if mass_per_L:
        m = mass_per_L * le / 420 * np.array([[156, 22 * le, 54, -13 * le], [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                              [54, 13 * le, 156, -22 * le], [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
    for e in range(ne):
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += k
        if mass_per_L:
            Mm[np.ix_(idx, idx)] += m
    F = np.zeros(2 * nn)
    for node, dof, val in loads:
        F[2 * node + dof] += val
    fixed = sorted({2 * nd + d for nd, ds in bcs.items() for d in ds})
    free = [i for i in range(2 * nn) if i not in fixed]
    u = np.zeros(2 * nn)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    Mend = []
    for e in range(ne):
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        f = k @ u[idx]
        Mend += [abs(f[1]), abs(f[3])]
    out = dict(w=u[0::2], th=u[1::2], Mmax=max(Mend))
    if mass_per_L:
        lam, _ = eigh(K[np.ix_(free, free)], Mm[np.ix_(free, free)])
        out["f"] = np.sqrt(lam) / (2 * np.pi)
    return out


def fe_beam_uniform_loads(L, n_el, W):
    """Consistent nodal loads for a uniform total load W."""
    le = L / n_el; q = W / L; loads = []
    for e in range(n_el):
        loads += [(e, 0, q * le / 2), (e, 1, q * le ** 2 / 12), (e + 1, 0, q * le / 2), (e + 1, 1, -q * le ** 2 / 12)]
    return loads


def axisym_plate_uniform(a, D, nu, q, clamped):
    """Axisymmetric plate by slope BVP: phi'' + phi'/r - phi/r^2 = Q/D, Q = q r / 2 (Timoshenko eq. 52 form).
    phi(0) = 0; clamped phi(a) = 0, SS M_r(a) = 0 -> phi' + nu phi / a = 0. Returns |w(0)|."""
    def f(r, y):
        p, p1 = y
        return np.vstack([p1, q * r / (2 * D) - p1 / r + p / r ** 2])
    def bc(ya, yb):
        return np.array([ya[0], yb[0] if clamped else yb[1] + nu * yb[0] / a])
    r = np.linspace(1e-6 * a, a, 500)
    sol = solve_bvp(f, bc, r, np.zeros((2, r.size)), tol=1e-10, max_nodes=200000)
    rr = np.linspace(1e-6 * a, a, 20001)
    return abs(np.trapezoid(sol.sol(rr)[0], rr))


def rotating_disk_bvp(rho, nu, E, omega, Ri, Ro):
    """Plane-stress rotating annular disk via displacement ODE u'' + u'/r - u/r^2 = -(1-nu^2) rho w^2 r / E.
    Returns hoop stress at Ri and radial stress at mid radius."""
    def f(r, y):
        u, u1 = y
        return np.vstack([u1, -(1 - nu * nu) * rho * omega ** 2 * r / E - u1 / r + u / r ** 2])
    def sr(r, u, u1):
        return E / (1 - nu * nu) * (u1 + nu * u / r)
    def bc(ya, yb):
        return np.array([sr(Ri, ya[0], ya[1]), sr(Ro, yb[0], yb[1])])
    r = np.linspace(Ri, Ro, 300)
    sol = solve_bvp(f, bc, r, np.zeros((2, r.size)), tol=1e-10)
    u, u1 = sol.sol(Ri)
    st = E / (1 - nu * nu) * (u / Ri + nu * u1)
    rm = 0.5 * (Ri + Ro); um, um1 = sol.sol(rm)
    return st, sr(rm, um, um1)


def integrate_impact(m, v0, force, t_max, rtol=1e-10):
    """m x'' = -force(x, x') from x=0, x'=v0 until separation. Returns peak x, peak force, contact time, exit v."""
    def rhs(t, y):
        return [y[1], -force(y[0], y[1]) / m]
    def sep(t, y):
        return y[0] if t > 1e-12 else 1.0
    sep.terminal = True; sep.direction = -1
    sol = solve_ivp(rhs, [0, t_max], [0.0, v0], events=sep, rtol=rtol, atol=1e-16, max_step=t_max / 2000,
                    dense_output=True)
    t = np.linspace(0, sol.t[-1], 4000); x = sol.sol(t)[0]; xd = sol.sol(t)[1]
    F = np.array([force(a, b) for a, b in zip(x, xd)])
    return x.max(), F.max(), sol.t[-1], sol.y[1, -1]

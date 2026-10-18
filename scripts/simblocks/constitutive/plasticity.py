"""1-D plasticity kernels for springs, hinges and dents (njit-safe).

Use any of these on a generalized (force, displacement) or (moment, rotation) or (stress, strain) pair.
State is passed in and returned so the loop owns it (no hidden globals):

  tabular_return        : full stress-strain-curve hardening (the replacement for bilinear models)
  linear_hardening_return : textbook radial return, isotropic + kinematic hardening
  lag_return            : EXACT reproduction of the update in a user-supplied impact sim, kept so old
                          results can be reproduced; it is a lag, not true hardening
"""
import math

from ..registry import block
from .._jit import kernel


@block("constitutive", "lookup", "table", "interp", kind="kernel", source="piecewise linear")
@kernel
def interp_clamped(x, xs, ys):
    """Piecewise-linear lookup with end clamping (ys[0] below, linear extrapolation of the last segment above)."""
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        s = (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2])
        return ys[n - 1] + s * (x - xs[n - 1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


@block("constitutive", "plasticity", "spring_1d", "tabular_hardening", kind="kernel",
       source="radial return with tabulated isotropic hardening (Simo & Hughes ch. 1)",
       tags=("stress strain curve", "hinge", "dent", "return map"))
@kernel
def tabular_return(x, xp, acc, k, acc_tab, F_tab, one_sided):
    """Elastic-plastic spring with isotropic hardening from a table F_y(acc).
    x      : current total displacement (or strain, rotation)
    xp     : plastic displacement from last step;  acc: accumulated |plastic| from last step
    k      : elastic stiffness;  acc_tab/F_tab: yield force vs accumulated plastic displacement
    one_sided : 1 -> compression-only contact style (force >= 0, yields only in +x); 0 -> symmetric
    Returns (F, xp_new, acc_new)."""
    Ftr = k * (x - xp)
    if one_sided == 1 and Ftr <= 0.0:
        return 0.0, xp, acc
    Fy = interp_clamped(acc, acc_tab, F_tab)
    if abs(Ftr) <= Fy:
        return Ftr, xp, acc
    sgn = 1.0 if Ftr > 0.0 else -1.0
    # solve Fy(acc + d) + k d = |Ftr| for d >= 0 (monotonic, bisection + secant polish)
    lo, hi = 0.0, (abs(Ftr) - Fy) / k
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = interp_clamped(acc + mid, acc_tab, F_tab) + k * mid - abs(Ftr)
        if g > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15 * (1.0 + abs(x)):
            break
    d = 0.5 * (lo + hi)
    xp_new = xp + sgn * d
    return k * (x - xp_new), xp_new, acc + d


@kernel
def tabular_return_rate(x, xp, acc, k, acc_tab, F_tab, one_sided, f):
    """tabular_return with the yield table scaled by f >= 1 this step (strain-rate hardening: the dynamic yield)."""
    Ftr = k * (x - xp)
    if one_sided == 1 and Ftr <= 0.0:
        return 0.0, xp, acc
    Fy = f * interp_clamped(acc, acc_tab, F_tab)
    if abs(Ftr) <= Fy:
        return Ftr, xp, acc
    sgn = 1.0 if Ftr > 0.0 else -1.0
    lo, hi = 0.0, (abs(Ftr) - Fy) / k
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = f * interp_clamped(acc + mid, acc_tab, F_tab) + k * mid - abs(Ftr)
        if g > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15 * (1.0 + abs(x)):
            break
    d = 0.5 * (lo + hi)
    xp_new = xp + sgn * d
    return k * (x - xp_new), xp_new, acc + d


@block("constitutive", "plasticity", "spring_1d", "linear_hardening", kind="kernel",
       source="Simo & Hughes 1998, Box 1.4", tags=("bilinear", "kinematic", "Bauschinger"))
@kernel
def linear_hardening_return(x, xp, acc, back, k, Fy0, H_iso, H_kin):
    """Bilinear elastic-plastic spring, combined isotropic (H_iso) and kinematic (H_kin) hardening.
    Hardening moduli are in force per plastic displacement. Returns (F, xp, acc, back)."""
    Ftr = k * (x - xp)
    xi = Ftr - back
    f = abs(xi) - (Fy0 + H_iso * acc)
    if f <= 0.0:
        return Ftr, xp, acc, back
    sgn = 1.0 if xi > 0.0 else -1.0
    dg = f / (k + H_iso + H_kin)
    xp += sgn * dg
    back += sgn * H_kin * dg
    acc += dg
    return k * (x - xp), xp, acc, back


@block("constitutive", "plasticity", "spring_1d", "legacy_lag", kind="kernel",
       source="LEGACY: user-supplied hinged-plate impact sim (verbatim, reproduction only; has a units bug in bend use)",
       tags=("legacy", "reproduce old results"))
@kernel
def lag_return(x, set_prev, y0, hard_coef, ratio, symmetric):
    """Original permanent-set update from the legacy hinged-plate sim. x = total deformation measure.
    dynamic yield = y0 + hard_coef * |set_prev|; if exceeded, set += excess * (1 - ratio).
    dent: symmetric=0, hard_coef = kp/(k-kp), ratio = kp/k.  bend: symmetric=1, hard_coef = kp, ratio = kp/k.
    Returns new set. Force is then k * (x - set) (clip >= 0 for the dent)."""
    dyn = y0 + hard_coef * abs(set_prev)
    e = x - set_prev
    if symmetric == 1:
        if abs(e) > dyn:
            sgn = 1.0 if e > 0.0 else -1.0
            return set_prev + (e - sgn * dyn) * (1.0 - ratio)
        return set_prev
    if e > dyn:
        return set_prev + (e - dyn) * (1.0 - ratio)
    return set_prev


@block("constitutive", "plasticity", "curve", "to_spring_table", kind="builder",
       source="sigma = F/A, eps = x/L scaling", tags=("stress strain curve", "bar", "tie"))
def curve_to_spring_table(curve, A, L, extend_to=0.5):
    """Convert a Curve's hardening table to a (k, acc_tab, F_tab) for an axial bar of area A, length L."""
    ep, sp = curve.plastic(extend_to=extend_to)
    return curve.E * A / L, ep * L, sp * A


@block("constitutive", "plasticity", "hinge", "to_spring_table", kind="builder",
       source="moment-rotation from moment-curvature", tags=("plastic hinge", "bending"))
def hinge_to_spring_table(hinge):
    """From structures.sections.hinge_table output to tabular_return inputs (k, acc_tab, M_tab).
    The elastic part of theta is removed so the table is yield moment vs plastic rotation."""
    import numpy as np
    th, M, k0 = hinge["theta"], hinge["M"], hinge["k0"]
    thp = th - M / k0
    idx = np.nonzero(thp > 1e-12 * max(th[-1], 1e-30))[0]
    if len(idx) == 0:
        return k0, np.array([0.0, 1.0]), np.array([M[-1], M[-1]])
    thp_t = np.concatenate([[0.0], thp[idx]])
    M_t = np.concatenate([[M[idx[0]]], M[idx]])
    out_t, out_m = [thp_t[0]], [M_t[0]]
    for a, b in zip(thp_t[1:], M_t[1:]):
        if a > out_t[-1]:
            out_t.append(a); out_m.append(max(b, out_m[-1]))
    return k0, np.array(out_t), np.array(out_m)


@block("constitutive", "plasticity", "spring_1d", "series_tables", kind="builder",
       source="compliance addition at equal force (two elastoplastic springs in series, isotropic hardening)",
       tags=("series", "dent", "indentation", "local", "combine", "elastoplastic spring"))
def elastoplastic_series(a, b, n=400):
    """Two elastoplastic springs in series (a tooth's surface indentation and the sheet's local dishing, a pad on a
    bracket): one table with the same force at every deflection. k = 1 / (1/k_a + 1/k_b); it yields at the weaker
    one's first yield; the plastic travel at force F is the sum of each spring's own plastic travel at F. Once one
    of them stops hardening (a flat end to its table) the pair flows at that force. a, b: dicts with k, acc_tab,
    F_tab (from any elastoplastic builder). Returns dict(k, acc_tab, F_tab, F_cap)."""
    import numpy as np

    def inv(t, F):
        """Plastic travel of table t at force F (0 below its yield; the far end of a flat stretch)."""
        x = np.asarray(t["acc_tab"], float); y = np.maximum.accumulate(np.asarray(t["F_tab"], float))
        if F <= y[0]:
            return 0.0
        if F >= y[-1]:
            return float(x[-1])
        j = int(np.searchsorted(y, F, side="right"))
        y0, y1 = y[j - 1], y[j]
        return float(x[j - 1] + (x[j] - x[j - 1]) * (F - y0) / (y1 - y0)) if y1 > y0 else float(x[j])
    ka, kb = float(a["k"]), float(b["k"])
    Fa, Fb = np.asarray(a["F_tab"], float), np.asarray(b["F_tab"], float)
    F0 = min(Fa[0], Fb[0]); Fcap = min(Fa.max(), Fb.max())
    Fs = np.unique(np.concatenate([np.linspace(F0, Fcap, n), Fa[(Fa > F0) & (Fa < Fcap)], Fb[(Fb > F0) & (Fb < Fcap)]]))
    acc = np.array([inv(a, F) + inv(b, F) for F in Fs])
    acc = np.maximum.accumulate(acc - acc[0])
    keep = np.concatenate([[True], np.diff(acc) > 0])            # the tabular return needs travel that grows
    return dict(k=1.0 / (1.0 / ka + 1.0 / kb), acc_tab=np.ascontiguousarray(acc[keep]),
                F_tab=np.ascontiguousarray(Fs[keep]), F_cap=float(Fcap))

"""Moment-curvature from a full stress-strain curve (replaces bilinear 'yield + plastic_k' bending).

Plane sections remain plane: eps(y) = kappa * y. M(kappa) = integral sigma(eps(y)) y w(y) dy.
Symmetric tension/compression is assumed (true for metals; NOT for composites, use CLT).
"""
import numpy as np

from ...registry import block
from .properties import _chord


@block("structures", "sections", "rectangle", "moment_curvature", kind="builder",
       source="Euler-Bernoulli fiber integration", tags=("stress strain curve", "plastic hinge"))
def moment_curvature_rect(curve, b, h, kappa_max=None, n=120, n_fiber=400, measure="true"):
    """M(kappa) table for a b x h rectangle from a Curve (simblocks.materials.Curve)."""
    w = lambda y: np.full_like(y, b)
    return _mk(curve, w, -h / 2, h / 2, kappa_max, n, n_fiber, measure)


@block("structures", "sections", "polygon", "moment_curvature", kind="builder",
       source="Euler-Bernoulli fiber integration", tags=("drawing", "plastic hinge"))
def moment_curvature_polygon(curve, xy, kappa_max=None, n=120, n_fiber=400, measure="true"):
    """M(kappa) table for an arbitrary polygon section (bending about its centroidal x axis)."""
    from .properties import polygon_section
    P = np.asarray(xy, float)
    cy = polygon_section(P)["cy"]
    P = P - [0.0, cy]
    w = lambda y: np.array([_chord(P, yy) for yy in y])
    return _mk(curve, w, P[:, 1].min(), P[:, 1].max(), kappa_max, n, n_fiber, measure)


def _mk(curve, wfun, ymin, ymax, kappa_max, n, n_fiber, measure):
    """M(kappa) by fibre integration with the neutral axis placed where the axial force is zero.

    measure="true" (default, finite strain): a fibre at y stretches by lam = 1 + kappa (y - y0); it carries the TRUE
    stress at its log strain on its CURRENT area A0 / lam (plastic flow keeps volume), so tension and compression
    fibres differ and the neutral axis y0 moves to keep N = 0. In tension that force is exactly sigma_eng A0.
    measure="eng": engineering stress at engineering strain on the original area (the small-strain textbook form,
    symmetric in tension and compression). Either way an elastic-perfectly-plastic section tends to Mp = sy Z."""
    true = measure == "true"
    es, ss = curve.true if true else (curve.eng_strain, curve.eng_stress)
    c = max(abs(ymin), abs(ymax))
    if kappa_max is None:
        kappa_max = (np.expm1(es[-1]) if true else es[-1]) / c
    y = np.linspace(ymin, ymax, n_fiber + 1)
    ym = 0.5 * (y[1:] + y[:-1]); dy = np.diff(y)
    wy = wfun(ym) * dy

    def fibres(k, y0):
        e = k * (ym - y0)
        if true:
            lam = np.maximum(1.0 + e, 1e-6)
            el = np.log(lam)
            return np.sign(el) * np.interp(np.abs(el), es, ss, right=ss[-1]) * wy / lam
        return np.sign(e) * np.interp(np.abs(e), es, ss, right=ss[-1]) * wy

    kap = np.concatenate([[0.0], np.geomspace(kappa_max * 1e-4, kappa_max, n - 1)])
    M = np.empty(n); Y0 = np.zeros(n)
    for i, k in enumerate(kap):
        if k == 0.0:
            M[i] = 0.0; continue
        lo, hi = ymin, ymax                         # N(y0) falls as the neutral axis rises: bisect for N = 0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fibres(k, mid).sum() > 0.0:
                lo = mid
            else:
                hi = mid
        y0 = 0.5 * (lo + hi); F = fibres(k, y0)
        M[i] = float(np.sum(F * (ym - y0))); Y0[i] = y0
    return dict(kappa=kap, M=M, c=c, EI0=M[1] / kap[1] if kap[1] > 0 else np.nan, y_neutral=Y0)


@block("structures", "sections", "general", "moment_rotation", kind="builder",
       source="lumped plastic hinge, theta = kappa * L_hinge", tags=("plastic hinge", "spring table"))
def hinge_table(mk, L_hinge):
    """Convert M(kappa) to a rotational-spring M(theta) table over hinge length L_hinge
    (common choice: L_hinge ~ section depth h). Use with constitutive.tabular kernels."""
    return dict(theta=mk["kappa"] * L_hinge, M=mk["M"], k0=mk["EI0"] / L_hinge)

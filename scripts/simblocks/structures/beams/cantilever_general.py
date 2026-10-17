"""One cantilever block for any mix of lateral loads: tip force and moment, a uniform load, point forces and
moments anywhere along it, and a distributed load (force and/or moment per length) from a CSV table.

Euler-Bernoulli bending, fixed at x = 0, free at x = L, small deflection. The moment along the beam is built from
every load (superposition is exact for a linear beam), then integrated twice for slope and deflection."""
import csv
import math
import os

import numpy as np

from ...registry import block


def _read_dist(path, x_unit, unit, m_unit):
    from ...compiler.units import unit_factor
    p = path
    if not os.path.isabs(p):
        p = os.path.join(os.environ.get("SIMBLOCKS_MODEL_DIR") or os.getcwd(), p)
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#"))]
    num = []
    for r in rows:
        try:
            num.append([float(c) for c in r if c.strip() != ""])
        except ValueError:
            continue                                   # a header row
    if not num:
        raise ValueError(f"no numbers in the CSV {path}")
    fx, fw = unit_factor(x_unit), unit_factor(unit)
    fm = unit_factor(m_unit) if m_unit else 1.0
    x = np.array([r[0] for r in num]) * fx
    w = np.array([r[1] if len(r) > 1 else 0.0 for r in num]) * fw
    m = np.array([r[2] if len(r) > 2 else 0.0 for r in num]) * fm
    if np.any(np.diff(x) <= 0):
        raise ValueError("the distribution's x must increase down the table")
    return x, w, m


_KINDS = ("point", "moment", "uniform", "linear", "csv")


def _case_loads(x, L, load, value, a, b, value_b, csv_file, csv_x_unit, csv_unit, csv_m_unit, n):
    """Shear V(x) and moment M(x) from one load case (everything beyond x, cantilever fixed at 0)."""
    kind = str(load).strip().lower()
    V = np.zeros_like(x); M = np.zeros_like(x)
    if kind not in _KINDS:
        raise ValueError(f"load is one of {', '.join(_KINDS)} (got {load!r})")
    if kind == "csv":
        if not csv_file:
            raise ValueError("load: csv needs csv_file")
        xd, wd, md = _read_dist(str(csv_file), csv_x_unit, csv_unit, csv_m_unit)
        return _dist(x, L, xd, wd, md, n)
    v = float(value or 0.0)
    if kind in ("point", "moment"):
        aa = L if a is None else float(a)
        if not 0 <= aa <= L:
            raise ValueError(f"the {kind} at {aa} m is off the beam (0 to {L} m)")
        on = x <= aa
        if kind == "point":
            V += v * on; M += v * (aa - x) * on
        else:
            M += v * on
        return V, M
    aa = 0.0 if a is None else float(a); bb = L if b is None else float(b)
    if not 0 <= aa < bb <= L:
        raise ValueError(f"a distributed load runs from a to b with 0 <= a < b <= L (got {aa} to {bb} m)")
    vb = v if (kind == "uniform" or value_b is None) else float(value_b)
    return _dist(x, L, np.array([aa, bb]), np.array([v, vb]), np.zeros(2), n, inside_only=True)


def _dist(x, L, xd, wd, md, n, inside_only=False):
    fine = np.linspace(0.0, L, 4 * int(n) + 1)
    wf = np.interp(fine, xd, wd, left=0.0, right=0.0); mf = np.interp(fine, xd, md, left=0.0, right=0.0)
    if inside_only:
        out = (fine < xd[0]) | (fine > xd[-1]); wf[out] = 0.0
    dx = fine[1] - fine[0]
    cw = np.concatenate([[0.0], np.cumsum(0.5 * (wf[1:] + wf[:-1]) * dx)])
    cws = np.concatenate([[0.0], np.cumsum(0.5 * (wf[1:] * fine[1:] + wf[:-1] * fine[:-1]) * dx)])
    cm = np.concatenate([[0.0], np.cumsum(0.5 * (mf[1:] + mf[:-1]) * dx)])
    Wb = cw[-1] - np.interp(x, fine, cw); Sb = cws[-1] - np.interp(x, fine, cws); Mb = cm[-1] - np.interp(x, fine, cm)
    return Wb, Sb - x * Wb + Mb


@block("structures", "beams", "cantilever", "general", kind="builder",
       source="Euler-Bernoulli beam, fixed-free; moments by superposition, slope and deflection by integration "
              "(matches Roark 7e Table 8.1 cases 1a-1e and 2a-2e)",
       tags=("cantilever", "tip load", "end moment", "uniform load", "distributed load", "triangular load", "CSV",
             "deflection", "bending moment", "beam"))
def cantilever(L, load="point", value=0.0, a=None, b=None, value_b=None, csv_file=None, more_loads=None,
               E=None, I=None, EI=None, mat=None, sec=None, c=None,
               csv_x_unit="m", csv_unit="N/m", csv_m_unit="N m/m", n=801):
    """Cantilever (fixed at x = 0, free at x = L) under one load case, or several added together.

    L          length (m)
    load       the load case:
                 point    a lateral force value (N) at a (default a = L, the free end)
                 moment   a moment value (N m) at a (default the free end)
                 uniform  a load value (N/m) spread evenly from a to b (default the whole length)
                 linear   a load spread from a to b, value (N/m) at a rising or falling to value_b at b
                 csv      a distributed load from csv_file: x, force per length, optional moment per length
    value, a, b, value_b   the case's numbers (see load)
    csv_file   the table for load: csv (path relative to the model's folder); csv_x_unit, csv_unit, csv_m_unit
               are its column units (default m, N/m, N m/m); linear between rows, zero outside
    more_loads more cases to add: [{load: point, value: 20, a: 0.1}, {load: uniform, value: 50, a: 0, b: 0.2}]
    E, I       Young's modulus (Pa) and second moment of area (m^4); or EI (N m^2); or mat and sec
    c          neutral axis to the outer fibre (m) for stress (from sec if given)
    Positive loads all bend the beam the same (positive) way.

    Returns dict(tip_deflection, tip_slope, root_moment, root_shear, max_moment, x_max_moment, max_stress,
    tip_stiffness, x, deflection, slope, moment, shear) in SI."""
    from ...materials import get
    if EI is None:
        if E is None and mat is not None:
            E = (get(mat) if isinstance(mat, str) else mat).E
        if I is None and sec is not None:
            I = float(sec["Ix"])
        if E is None or I is None:
            raise ValueError("cantilever needs EI, or E and I, or a material and a section")
        EI = float(E) * float(I)
    if c is None and sec is not None:
        c = sec.get("c")
    L = float(L); EI = float(EI)
    if L <= 0 or EI <= 0:
        raise ValueError("L and EI must be positive")
    x = np.linspace(0.0, L, int(n))
    V = np.zeros_like(x); M = np.zeros_like(x)
    cases = [dict(load=load, value=value, a=a, b=b, value_b=value_b, csv_file=csv_file)]
    for extra in (more_loads or []):
        if not isinstance(extra, dict) or "load" not in extra:
            raise ValueError("more_loads is a list of {load: ..., value: ..., a: ..., b: ...}")
        bad = set(extra) - {"load", "value", "a", "b", "value_b", "csv_file"}
        if bad:
            raise ValueError("unknown key(s) in more_loads: " + ", ".join(sorted(bad)))
        cases.append(dict(dict(value=0.0, a=None, b=None, value_b=None, csv_file=None), **extra))
    for cs in cases:
        dV, dM = _case_loads(x, L, cs["load"], cs["value"], cs["a"], cs["b"], cs["value_b"], cs["csv_file"],
                             csv_x_unit, csv_unit, csv_m_unit, n)
        V += dV; M += dM
    k = M / EI; dx = x[1] - x[0]
    th = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * dx)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (th[1:] + th[:-1]) * dx)])
    i = int(np.argmax(np.abs(M)))
    out = dict(tip_deflection=float(y[-1]), tip_slope=float(th[-1]), root_moment=float(M[0]), root_shear=float(V[0]),
               max_moment=float(M[i]), x_max_moment=float(x[i]), tip_stiffness=3.0 * EI / L ** 3, EI=EI,
               x=x, deflection=y, slope=th, moment=M, shear=V)
    I_ = float(I) if I else (EI / float(E) if E else None)
    out["max_stress"] = float(abs(M[i]) * float(c) / I_) if (c and I_) else math.nan     # M c / I at the peak moment
    return out

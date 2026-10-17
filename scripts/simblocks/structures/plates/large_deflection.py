"""Membrane theory: plates and membranes deflecting far enough that they stretch as well as bend.

Every block here is a closed-form result (an energy-method load-deflection law, an exact membrane solution's
constants, or a rigid-plastic hinge-line result) solved with a root-find where the formula is implicit:
  circular_plate_large_deflection   bending + membrane, q a^4/(E t^4) = k1 (w/t) + k3 (w/t)^3
  membrane_circular                 a pure membrane (no bending stiffness): Hencky (pressure), Schwerin (centre load)
  plate_strip_large_deflection      a long plate in cylindrical bending with membrane tension (Timoshenko, exact)
  plate_plastic_membrane            rigid-plastic plate under a central load: collapse then membrane hardening (Jones)
"""
import math

import numpy as np

from ...registry import block

SRC_RITZ = ("Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, sec. 96-97 (energy method), with three "
            "in-plane terms; membrane term of held edges set to the exact Hencky / Schwerin limit")
SRC_HENCKY = "Hencky (1915) pressurised membrane, Fichter NASA TP-3658 (1997) corrected constants; Schwerin centre load"
SRC_STRIP = "Timoshenko & Woinowsky-Krieger sec. 2-3: cylindrical bending of a long plate with membrane tension"
SRC_JONES = "Jones, Structural Impact (1989) ch. 7; Jones, Lat. Am. J. Solids Struct. 10 (2013) 767, eqs. 4, 5, 8, 9"

# Exact pure-membrane constants (held edge) against Poisson's ratio:
#   pressure q:   w0 = Cw a (q a / E t)^(1/3),  sigma = Cs (E q^2 a^2 / t^2)^(1/3)   (centre; radial and hoop at the edge)
#   centre load F: w0 = Cw a (F / E t a)^(1/3),  sigma at the edge = Cs (E F^2 / a^2 t^2)^(1/3)
# columns: nu, pressure Cw, centre, edge radial, edge hoop | point Cw, edge radial, edge hoop
_HENCKY = np.array([
    [0.00, 0.71832, 0.40510, 0.28937, 0.00000, 1.05106, 0.19736, 0.00000],
    [0.10, 0.69890, 0.41218, 0.30178, 0.03018, 1.03385, 0.20831, 0.02083],
    [0.20, 0.67744, 0.42068, 0.31611, 0.06322, 1.01456, 0.22118, 0.04424],
    [0.25, 0.66579, 0.42561, 0.32417, 0.08104, 1.00397, 0.22851, 0.05713],
    [0.30, 0.65344, 0.43110, 0.33296, 0.09989, 0.99265, 0.23658, 0.07097],
    [0.33, 0.64566, 0.43471, 0.33863, 0.11175, 0.98547, 0.24183, 0.07980],
    [0.35, 0.64030, 0.43726, 0.34259, 0.11991, 0.98049, 0.24552, 0.08593],
    [0.40, 0.62626, 0.44422, 0.35323, 0.14129, 0.96736, 0.25549, 0.10220],
    [0.45, 0.61117, 0.45216, 0.36508, 0.16429, 0.95309, 0.26673, 0.12003],
    [0.50, 0.59488, 0.46133, 0.37841, 0.18921, 0.93748, 0.27953, 0.13976]])


def _hencky(nu, col):
    if not 0.0 <= nu <= 0.5:
        raise ValueError(f"Poisson's ratio {nu} is outside 0 to 0.5")
    return float(np.interp(nu, _HENCKY[:, 0], _HENCKY[:, col]))


# Energy-method coefficients (plate with a = t = E = 1): load = k1 x + k3 x^3 with x = w0 / t; membrane stress per x^2
# (sm_*), bending stress per x (sb_*), at the centre (_c) and the edge (_e), radial (_r) and hoop (_t)
_RITZ = {
    "pressure|clamped|held": dict(
        k1=lambda nu: -16/(3*nu**2 - 3),
        k3=lambda nu: (2/800415)*(342831*nu**2 - 529610*nu - 896585)/(nu**2 - 1),
        sm_r_c=lambda nu: (1/1386)*(723*nu - 1585)/(nu - 1),
        sm_r_e=lambda nu: (1/693)*(-211*nu - 415)/(nu**2 - 1),
        sm_t_c=lambda nu: (1/1386)*(723*nu - 1585)/(nu - 1),
        sm_t_e=lambda nu: -nu*(211*nu + 415)/(693*nu**2 - 693),
        sb_r_e=lambda nu: 4/(nu**2 - 1),
        sb_r_c=lambda nu: -2/(nu - 1),
        sb_t_e=lambda nu: 4*nu/(nu**2 - 1),
        sb_t_c=lambda nu: -2/(nu - 1),
    ),
    "pressure|clamped|free": dict(
        k1=lambda nu: -16/(3*nu**2 - 3),
        k3=lambda nu: (2/6615)*(2791*nu**2 + 160*nu - 3095)/(nu**2 - 1),
        sm_r_c=lambda nu: (1/126)*(89*nu - 137)/(nu - 1),
        sm_r_e=lambda nu: (4/63)*(-nu - 2)/(nu**2 - 1),
        sm_t_c=lambda nu: (1/126)*(89*nu - 137)/(nu - 1),
        sm_t_e=lambda nu: (1/63)*(-nu*(25*nu + 8) + 21)/(nu**2 - 1),
        sb_r_e=lambda nu: 4/(nu**2 - 1),
        sb_r_c=lambda nu: -2/(nu - 1),
        sb_t_e=lambda nu: 4*nu/(nu**2 - 1),
        sb_t_c=lambda nu: -2/(nu - 1),
    ),
    "pressure|simply_supported|held": dict(
        k1=lambda nu: -16/(3*nu**2 + 12*nu - 15),
        k3=lambda nu: (2/800415)*(342831*nu**5 + 4768907*nu**4 + 19870230*nu**3 - 16699626*nu**2 - 331850965*nu - 709519665)/(nu**5 + 21*nu**4 + 158*nu**3 + 470*nu**2 + 225*nu - 875),
        sm_r_c=lambda nu: (1/1386)*(723*nu**3 + 3337*nu**2 + 281*nu - 24509)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sm_r_e=lambda nu: (1/693)*(-211*nu**2 - 2518*nu - 7587)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sm_t_c=lambda nu: (1/1386)*(723*nu**3 + 3337*nu**2 + 281*nu - 24509)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sm_t_e=lambda nu: -nu*(211*nu**2 + 2518*nu + 7587)/(693*nu**3 + 6237*nu**2 + 10395*nu - 17325),
        sb_r_e=lambda nu: 0,
        sb_r_c=lambda nu: 2*(-nu - 3)/(nu**2 + 4*nu - 5),
        sb_t_e=lambda nu: 4/(nu + 5),
        sb_t_c=lambda nu: 2*(-nu - 3)/(nu**2 + 4*nu - 5),
    ),
    "pressure|simply_supported|free": dict(
        k1=lambda nu: -16/(3*nu**2 + 12*nu - 15),
        k3=lambda nu: (2/6615)*(2791*nu**5 + 43357*nu**4 + 253390*nu**3 + 607714*nu**2 + 144715*nu - 1096895)/(nu**5 + 21*nu**4 + 158*nu**3 + 470*nu**2 + 225*nu - 875),
        sm_r_c=lambda nu: (1/126)*(89*nu**3 + 485*nu**2 - 113*nu - 1517)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sm_r_e=lambda nu: (4/63)*(-nu**2 - 12*nu + 43)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sm_t_c=lambda nu: (1/126)*(89*nu**3 + 485*nu**2 - 113*nu - 1517)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sm_t_e=lambda nu: (1/63)*(-25*nu**3 - 237*nu**2 - 311*nu + 693)/(nu**3 + 9*nu**2 + 15*nu - 25),
        sb_r_e=lambda nu: 0,
        sb_r_c=lambda nu: 2*(-nu - 3)/(nu**2 + 4*nu - 5),
        sb_t_e=lambda nu: 4/(nu + 5),
        sb_t_c=lambda nu: 2*(-nu - 3)/(nu**2 + 4*nu - 5),
    ),
    "point|clamped|held": dict(
        k1=lambda nu: -4*math.pi/(3*nu**2 - 3),
        k3=lambda nu: (1/2917215000)*math.pi*(859233241*nu**2 - 726861486*nu - 1591570311)/(nu**2 - 1),
        sm_r_c=lambda nu: (1/17640)*(17551*nu - 24513)/(nu - 1),
        sm_r_e=lambda nu: (1/44100)*(-10387*nu - 12099)/(nu**2 - 1),
        sm_t_c=lambda nu: (1/17640)*(17551*nu - 24513)/(nu - 1),
        sm_t_e=lambda nu: -nu*(10387*nu + 12099)/(44100*nu**2 - 44100),
        sb_r_e=lambda nu: 2/(nu**2 - 1),
        sb_t_e=lambda nu: 2*nu/(nu**2 - 1),
    ),
    "point|clamped|free": dict(
        k1=lambda nu: -4*math.pi/(3*nu**2 - 3),
        k3=lambda nu: (1/1215000)*math.pi*(357049*nu**2 + 12696*nu - 400929)/(nu**2 - 1),
        sm_r_c=lambda nu: (1/600)*(553*nu - 369)/(nu - 1),
        sm_r_e=lambda nu: (1/450)*(23*nu - 129)/(nu**2 - 1),
        sm_t_c=lambda nu: (1/600)*(553*nu - 369)/(nu - 1),
        sm_t_e=lambda nu: (1/900)*(-nu*(179*nu + 258) + 225)/(nu**2 - 1),
        sb_r_e=lambda nu: 2/(nu**2 - 1),
        sb_t_e=lambda nu: 2*nu/(nu**2 - 1),
    ),
    "point|simply_supported|held": dict(
        k1=lambda nu: -4*math.pi/(3*nu**2 + 6*nu - 9),
        k3=lambda nu: (1/2917215000)*math.pi*(859233241*nu**5 + 6279376197*nu**4 + 12206236314*nu**3 - 23591798870*nu**2 - 127819257699*nu - 159675285471)/(nu**5 + 11*nu**4 + 42*nu**3 + 54*nu**2 - 27*nu - 81),
        sm_r_c=lambda nu: (1/17640)*(17551*nu**3 + 38729*nu**2 - 34835*nu - 126573)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sm_r_e=lambda nu: (1/44100)*(-10387*nu**2 - 65746*nu - 146919)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sm_t_c=lambda nu: (1/17640)*(17551*nu**3 + 38729*nu**2 - 34835*nu - 126573)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sm_t_e=lambda nu: -nu*(10387*nu**2 + 65746*nu + 146919)/(44100*nu**3 + 220500*nu**2 + 132300*nu - 396900),
        sb_r_e=lambda nu: 0,
        sb_t_e=lambda nu: 2/(nu + 3),
    ),
    "point|simply_supported|free": dict(
        k1=lambda nu: -4*math.pi/(3*nu**2 + 6*nu - 9),
        k3=lambda nu: (1/1215000)*math.pi*(357049*nu**5 + 2935083*nu**4 + 8650146*nu**3 + 9000670*nu**2 - 5784411*nu - 15192969)/(nu**5 + 11*nu**4 + 42*nu**3 + 54*nu**2 - 27*nu - 81),
        sm_r_c=lambda nu: (1/600)*(553*nu**3 + 1877*nu**2 - 65*nu - 2589)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sm_r_e=lambda nu: (1/450)*(23*nu**2 - 166*nu + 51)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sm_t_c=lambda nu: (1/600)*(553*nu**3 + 1877*nu**2 - 65*nu - 2589)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sm_t_e=lambda nu: (1/900)*(-179*nu**3 - 1457*nu**2 - 1473*nu + 2925)/(nu**3 + 5*nu**2 + 3*nu - 9),
        sb_r_e=lambda nu: 0,
        sb_t_e=lambda nu: 2/(nu + 3),
    ),
}


def _mat_E_nu(E, nu, mat):
    if mat is not None:
        from ...materials import get
        m = get(mat) if isinstance(mat, str) else mat
        E = m.E if E is None else E
        nu = (m.nu if getattr(m, "nu", None) else 0.3) if nu is None else nu
    if E is None:
        raise ValueError("give E (Pa) or a material")
    return float(E), float(0.3 if nu is None else nu)


def _solve_cubic(k1, k3, P):
    """Positive root of k1 x + k3 x^3 = P (monotone, so a short Newton from the larger of the two limits)."""
    if P <= 0:
        return 0.0
    x = min(P / k1, (P / k3) ** (1.0 / 3.0)) if k3 > 0 else P / k1
    for _ in range(60):
        f = k1 * x + k3 * x ** 3 - P; d = k1 + 3 * k3 * x * x
        dx = f / d; x -= dx
        if abs(dx) < 1e-14 * max(x, 1e-30):
            break
    return x


@block("structures", "plates", "circular", "large_deflection", kind="builder", source=SRC_RITZ,
       tags=("membrane", "large deflection", "von Karman", "plate stretching", "pressure", "point load", "diaphragm"))
def circular_plate_large_deflection(a, t, value, load="pressure", edge="clamped", inplane="held", E=None, nu=None,
                                    mat=None, n=60):
    """Circular plate (radius a, thickness t) whose deflection is large enough that it stretches as well as bends.
    load: pressure (value = q, Pa) or point (value = F at the centre, N). edge: clamped or simply_supported.
    inplane: held (the rim cannot move inward: membrane tension builds up) or free (it can slide in).
    Load-deflection law  load* = k1 (w0/t) + k3 (w0/t)^3,  load* = q a^4 / (E t^4) or F a^2 / (E t^4):
    k1 is the exact small-deflection stiffness; k3 the membrane stretching. Below w0 ~ t/2 bending dominates; past a
    few t it acts as a membrane.
    Returns dict(w0, x = w0/t, sigma_max (surface, bending + membrane), sigma_membrane_centre, sigma_membrane_edge,
    sigma_bending_centre, sigma_bending_edge, k_secant, k_tangent, k1, k3, w_tab, F_tab) in SI. F_tab is the total
    load (q pi a^2 for pressure) against w_tab; F_resist = -F_tab makes the plate a nonlinear spring on a force
    element whose measure grows as the plate deflects: csv {x: $plate.w_tab, y: $plate.F_resist, against: measure}
    (value sets the top of the table: give the largest load you expect, the end value holds beyond it)."""
    if load not in ("pressure", "point"):
        raise ValueError("load is pressure or point")
    if edge not in ("clamped", "simply_supported"):
        raise ValueError("edge is clamped or simply_supported")
    if inplane not in ("held", "free"):
        raise ValueError("inplane is held or free")
    E, nu = _mat_E_nu(E, nu, mat); a = float(a); t = float(t); value = float(value)
    if a <= 0 or t <= 0:
        raise ValueError("a and t must be positive")
    c = _RITZ[f"{load}|{edge}|{inplane}"]
    k1 = c["k1"](nu); k3 = c["k3"](nu)
    if inplane == "held":                 # the large-deflection end is the exact membrane solution
        k3 = 1.0 / _hencky(nu, 1 if load == "pressure" else 5) ** 3
    norm = a ** 4 / (E * t ** 4) if load == "pressure" else a ** 2 / (E * t ** 4)
    x = _solve_cubic(k1, k3, value * norm)
    w0 = x * t; s2 = E * (t / a) ** 2
    smc = s2 * x * x * c["sm_r_c"](nu) if "sm_r_c" in c else float("nan")
    sme = s2 * x * x * max(abs(c["sm_r_e"](nu)), abs(c["sm_t_e"](nu)))
    sbe = s2 * x * max(abs(c["sb_r_e"](nu)), abs(c["sb_t_e"](nu)))
    sbc = s2 * x * abs(c["sb_r_c"](nu)) if "sb_r_c" in c else float("inf")      # a point load's moment is singular
    sig = max(abs(smc) + (sbc if math.isfinite(sbc) else 0.0), sme + sbe)
    area = math.pi * a * a if load == "pressure" else 1.0
    xs = np.linspace(0.0, max(x, 1e-12), int(n))
    F_tab = (k1 * xs + k3 * xs ** 3) / norm * area
    kt = (k1 + 3 * k3 * x * x) / norm * area / t
    return dict(w0=w0, x=x, sigma_max=sig, sigma_membrane_centre=smc, sigma_membrane_edge=sme,
                sigma_bending_centre=sbc, sigma_bending_edge=sbe,
                k_secant=(value * area / w0) if w0 > 0 else kt, k_tangent=kt, k1=k1, k3=k3,
                w_tab=xs * t, F_tab=F_tab, F_resist=-F_tab)


@block("structures", "plates", "membrane", "circular", kind="builder", source=SRC_HENCKY,
       tags=("membrane", "diaphragm", "Hencky", "pressure", "point load", "film", "sheet"))
def membrane_circular(a, t, value, load="pressure", E=None, nu=None, mat=None):
    """A circular membrane with no bending stiffness (a thin sheet, film or diaphragm), edge held, under a pressure q
    (load: pressure, value in Pa) or a centre load F (load: point, value in N). Exact membrane solution:
      pressure: w0 = Cw a (q a / E t)^(1/3),   sigma_centre = Cs (E q^2 a^2 / t^2)^(1/3)
      point:    w0 = Cw a (F / E t a)^(1/3),   stress is infinite under an ideal point (give the edge value)
    Cw, Cs from Poisson's ratio (Cw = 0.653 at nu = 0.3, Fichter's corrected Hencky constant).
    Returns dict(w0, sigma_centre, sigma_edge_radial, sigma_edge_hoop, k_secant) in SI."""
    E, nu = _mat_E_nu(E, nu, mat); a = float(a); t = float(t); value = float(value)
    if load == "pressure":
        w0 = _hencky(nu, 1) * a * (value * a / (E * t)) ** (1.0 / 3.0)
        S = (E * value ** 2 * a ** 2 / t ** 2) ** (1.0 / 3.0)
        sc, sr, sh = _hencky(nu, 2) * S, _hencky(nu, 3) * S, _hencky(nu, 4) * S
        F = value * math.pi * a * a
    elif load == "point":
        w0 = _hencky(nu, 5) * a * (value / (E * t * a)) ** (1.0 / 3.0)
        S = (E * value ** 2 / (a * a * t * t)) ** (1.0 / 3.0)
        sc, sr, sh = float("inf"), _hencky(nu, 6) * S, _hencky(nu, 7) * S
        F = value
    else:
        raise ValueError("load is pressure or point")
    return dict(w0=w0, sigma_centre=sc, sigma_edge_radial=sr, sigma_edge_hoop=sh, k_secant=F / w0 if w0 > 0 else 0.0)


def _shm(z):
    """sinh(z) - z without cancellation for small z."""
    if abs(z) < 0.5:
        z2 = z * z; return z * z2 * (1 / 6 + z2 * (1 / 120 + z2 * (1 / 5040 + z2 * (1 / 362880 + z2 / 39916800))))
    return math.sinh(z) - z


def _chm(z):
    """cosh(z) - 1 without cancellation for small z."""
    return 2.0 * math.sinh(0.5 * z) ** 2


_GL = np.polynomial.legendre.leggauss(64)


def _strip_state(k, q, l, D, edge):
    """Closed-form shape of a long strip (span l, per unit width) under q with membrane tension S = D k^2, written
    so it stays accurate as the tension goes to zero. Returns (w0, half the integral of w'^2, M midspan, M edge)."""
    L = l / 2.0; u = k * L
    xg, wg = _GL; xi = 0.5 * L * (xg + 1.0); wts = 0.5 * L * wg; z = k * xi
    c = q / (D * k ** 3)
    if edge == "simply_supported":         # w' = c [sinh z - z cosh u] / cosh u
        br = np.array([_shm(zz) - zz * _chm(u) for zz in z]) / math.cosh(u)
        Mmid = q / (k * k) * _chm(u) / math.cosh(u); Medge = 0.0
    else:                                   # w' = c [u sinh z - z sinh u] / sinh u
        br = np.array([u * _shm(zz) - zz * _shm(u) for zz in z]) / math.sinh(u)
        Mmid = q / (k * k) * _shm(u) / math.sinh(u); Medge = q / (k * k) * (_shm(u) - u * _chm(u)) / math.sinh(u)
    wp = c * br
    w0 = -float(np.dot(wts, wp))           # w(L) = 0, so w(0) = -integral of w' from 0 to L
    stretch = float(np.dot(wts, wp * wp))  # (1/2) integral over the whole span = integral over the half span
    return w0, stretch, Mmid, Medge


@block("structures", "plates", "strip", "large_deflection", kind="builder", source=SRC_STRIP,
       tags=("membrane", "large deflection", "cylindrical bending", "long plate", "panel", "pressure"))
def plate_strip_large_deflection(l, t, q, edge="simply_supported", E=None, nu=None, mat=None):
    """A long plate (span l between two long edges, thickness t) under pressure q, bending into a cylinder, with both
    edges held so membrane tension S builds up as it deflects. Exact: the beam-column shape with tension S, and S set
    so the extension S l (1 - nu^2) / (E t) equals the arc-length gain (1/2) integral of w'^2 (a root-find for S).
    edge: simply_supported or clamped. Per unit width.
    Returns dict(w0, S (N/m), sigma_membrane, sigma_bending_max, sigma_max, u = (l/2) sqrt(S/D)) in SI."""
    if edge not in ("simply_supported", "clamped"):
        raise ValueError("edge is simply_supported or clamped")
    E, nu = _mat_E_nu(E, nu, mat); l = float(l); t = float(t); q = float(q)
    D = E * t ** 3 / (12.0 * (1.0 - nu * nu))
    if q <= 0:
        return dict(w0=0.0, S=0.0, sigma_membrane=0.0, sigma_bending_max=0.0, sigma_max=0.0, u=0.0)
    k_of = lambda u: 2.0 * u / l
    g = lambda u: D * k_of(u) ** 2 * l * (1.0 - nu * nu) / (E * t) - _strip_state(k_of(u), q, l, D, edge)[1]
    lo, hi = 1e-7, 1e4                    # u = (l/2) sqrt(S/D): from a nearly flat plate to a pure membrane
    a_, b_ = math.log(lo), math.log(hi)
    for _ in range(200):
        m = 0.5 * (a_ + b_)
        if g(math.exp(m)) < 0:
            a_ = m
        else:
            b_ = m
    u = math.exp(0.5 * (a_ + b_)); k = k_of(u); S = D * k * k
    w0, _, Mmid, Medge = _strip_state(k, q, l, D, edge)
    Mmax = max(abs(Mmid), abs(Medge))
    sm = S / t; sb = 6.0 * Mmax / t ** 2
    return dict(w0=w0, S=S, sigma_membrane=sm, sigma_bending_max=sb, sigma_max=sm + sb, u=u)


@block("structures", "plates", "plastic", "membrane_collapse", kind="builder", source=SRC_JONES,
       tags=("membrane", "rigid-plastic", "large deflection", "dent", "collapse", "armour", "point load", "punch"))
def plate_plastic_membrane(t, sy=None, shape="circular", edge="clamped", a=None, B=None, L=None, mat=None,
                           rate_factor=1.0, W_max=None, E=None, nu=None, n=60):
    """A plate pushed at its centre past collapse, where membrane stretching makes it stiffen as it deflects
    (rigid-plastic, axially restrained edges). Jones' result for a central load:
      P / Pc = 1 + W / t  (clamped),   P / Pc = 1 + 2 W / t  (simply supported)
    circular (radius a):              Pc = pi sy t^2 (clamped),  pi sy t^2 / 2 (simply supported)
    rectangular (2B x 2L, B <= L):    Pc = 2 sy t^2 (1 + b^2) / b (clamped), half that simply supported, b = B / L
    sy is the flow stress (Jones uses the yield stress; 0.618 sy gives a lower bound), times rate_factor.
    Returns an elastoplastic table {k, acc_tab, F_tab} plus Pc, M0 = sy t^2 / 4: k is the plate's small-deflection
    centre-load stiffness (circular exact; rectangular from the circle of equal area). acc_tab runs to W_max
    (default 10 t). Use it on the measure of the dent or the plate's centre."""
    if sy is None:
        if mat is None:
            raise ValueError("give sy (Pa) or a material")
        from ...materials import get
        m = get(mat) if isinstance(mat, str) else mat
        sy = m.sy
    E, nu = _mat_E_nu(E, nu, mat if E is None else None) if (E is not None or mat is not None) else (None, nu or 0.3)
    sy = float(sy) * float(rate_factor); t = float(t)
    if edge not in ("clamped", "simply_supported"):
        raise ValueError("edge is clamped or simply_supported")
    if shape == "circular":
        if a is None:
            raise ValueError("a circular plate needs a (radius)")
        Pc = math.pi * sy * t * t * (1.0 if edge == "clamped" else 0.5); R_eq = float(a)
    elif shape == "rectangular":
        if B is None or L is None:
            raise ValueError("a rectangular plate needs B and L (half-widths, 2B x 2L)")
        B, L = sorted((float(B), float(L))); b = B / L
        Pc = 2.0 * sy * t * t * (1.0 + b * b) / b * (1.0 if edge == "clamped" else 0.5)
        R_eq = math.sqrt(4.0 * B * L / math.pi)
    else:
        raise ValueError("shape is circular or rectangular")
    cm = 1.0 if edge == "clamped" else 2.0
    if E is not None:
        D = E * t ** 3 / (12.0 * (1.0 - nu * nu))
        k = 16.0 * math.pi * D / R_eq ** 2 * (1.0 if edge == "clamped" else (1.0 + nu) / (3.0 + nu))
    else:
        k = 1e3 * Pc / t                      # rigid-plastic: a stiff elastic start
    Wm = float(W_max) if W_max else 10.0 * t
    acc = np.linspace(0.0, Wm, int(n))
    F = Pc * (1.0 + cm * acc / t)
    return dict(k=float(k), acc_tab=np.ascontiguousarray(acc), F_tab=np.ascontiguousarray(F), Pc=float(Pc),
                M0=sy * t * t / 4.0, source=f"Jones rigid-plastic membrane, {shape} {edge}, sy {sy / 1e6:.4g} MPa")

"""Cross-section properties.

Every builder returns a dict with the same keys so sections are interchangeable:
  A    area                       Ix   second moment about the bending axis (x, horizontal)
  Iy   second moment about y      c    extreme-fiber distance for Ix
  S    elastic section modulus Ix/c          Z   plastic section modulus (about x)
  J    torsion constant (Saint-Venant)        tau_coef: tau_max = tau_coef * T
  kappa  Timoshenko shear coefficient
Units SI. The bending axis is x; "h" is always the dimension in the bending direction.
"""
import math
import numpy as np

from ...registry import block


def _sec(A, Ix, Iy, c, Z, J, tau_coef, kappa, **extra):
    d = dict(A=A, Ix=Ix, Iy=Iy, c=c, S=Ix / c, Z=Z, J=J, tau_coef=tau_coef, kappa=kappa)
    d.update(extra)
    return d


@block("structures", "sections", "rectangle", "properties", kind="builder",
       source="Roark 7e Table A.1 case 1; torsion Table 10.1 case 4", tags=("plate strip", "bar"))
def rectangle(b, h):
    """Solid rectangle, width b, depth h (bending direction). Torsion from Roark Table 10.1 case 4."""
    a, bb = max(b, h) / 2, min(b, h) / 2          # half-dims, a >= bb
    r = bb / a
    K = a * bb ** 3 * (16 / 3 - 3.36 * r * (1 - r ** 4 / 12))
    tau = 3 / (8 * a * bb ** 2) * (1 + 0.6095 * r + 0.8865 * r ** 2 - 1.8023 * r ** 3 + 0.9100 * r ** 4)
    return _sec(b * h, b * h ** 3 / 12, h * b ** 3 / 12, h / 2, b * h ** 2 / 4, K, tau, 5 / 6, b=b, h=h, Zy=h * b ** 2 / 4)


@block("structures", "sections", "given", "properties", kind="builder",
       source="user-supplied section properties (CAD, catalog or hand calc)", tags=("custom", "I", "second moment", "CAD"))
def given_section(I, c=None, h=None, A=float("nan"), Z=None, J=float("nan"), Iy=float("nan"), kappa=5 / 6, Zy=float("nan")):
    """Any section from its numbers, when it is not a circle or rectangle. Only I is required.
    I  second moment of area about the bending axis (stiffness needs only this)
    c  distance from the neutral axis to the longest (extreme) fibre, for max bending stress M c / I.
       Give c or the depth h (c = h/2 for a symmetric section). Left out: stress and yield results are NaN,
       stiffness and deflection still work.
    Z  plastic section modulus (optional): left out, Z = S = I/c, so no plastic reserve (conservative)
    A, J, Iy, kappa: only for blocks that use them (shear deflection, torsion, mass).
    Zy plastic modulus about the other (y) axis, for a 3D beam's plastic hinge (optional)."""
    I = float(I)
    if I <= 0:
        raise ValueError("I must be positive")
    if c is None and h is not None:
        c = float(h) / 2
    c = float("nan") if c is None else float(c)
    if c == c and c <= 0:
        raise ValueError("c (distance to the longest fibre) must be positive")
    S = I / c if c == c else float("nan")
    Zv = S if Z is None else float(Z)
    tau = c / J if (J == J and c == c) else float("nan")
    return dict(A=A, Ix=I, Iy=Iy, c=c, S=S, Z=Zv, J=J, tau_coef=tau, kappa=kappa, given=True, Zy=float(Zy))


@block("structures", "sections", "plate_strip", "properties", kind="builder",
       source="Roark 7e Table A.1 case 1 per unit width", tags=("plate", "sheet", "per width"))
def plate_per_width(t):
    """Unit-width strip of plate thickness t (all quantities per metre of width)."""
    return _sec(t, t ** 3 / 12, float("nan"), t / 2, t ** 2 / 4, t ** 3 / 3, 3 / t ** 2, 5 / 6, t=t)


@block("structures", "sections", "circle", "properties", kind="builder",
       source="Roark 7e Table A.1 case 15, Table 10.1 case 1", tags=("shaft", "pin", "rod"))
def circle(d):
    """Solid round of diameter d."""
    r = d / 2
    I = math.pi * d ** 4 / 64
    return _sec(math.pi * r * r, I, I, r, d ** 3 / 6, math.pi * d ** 4 / 32, 16 / (math.pi * d ** 3),
                6 / 7, d=d, Zy=d ** 3 / 6)


@block("structures", "sections", "tube", "properties", kind="builder",
       source="Roark 7e Table A.1 case 16, Table 10.1 case 10", tags=("shaft", "hollow"))
def tube(do, di):
    """Round tube, outer diameter do, inner di."""
    I = math.pi * (do ** 4 - di ** 4) / 64
    J = math.pi * (do ** 4 - di ** 4) / 32
    m = di / do
    kappa = 6 * (1 + m * m) ** 2 / (7 + 34 * m * m + 7 * m ** 4)   # Cowper 1966 (nu=0)
    return _sec(math.pi * (do ** 2 - di ** 2) / 4, I, I, do / 2, (do ** 3 - di ** 3) / 6, J,
                (do / 2) / J, kappa, do=do, di=di, Zy=(do ** 3 - di ** 3) / 6)


@block("structures", "sections", "rect_tube", "properties", kind="builder",
       source="Roark 7e Table A.1; torsion Bredt thin-wall Table 10.1 case 16", tags=("box", "hollow"))
def rect_tube(b, h, t):
    """Rectangular tube, outside b x h, uniform wall t. Torsion by Bredt thin-wall formula."""
    bi, hi = b - 2 * t, h - 2 * t
    Am = (b - t) * (h - t)
    J = 4 * Am ** 2 * t / (2 * ((b - t) + (h - t)))
    Z = b * h ** 2 / 4 - bi * hi ** 2 / 4
    return _sec(b * h - bi * hi, (b * h ** 3 - bi * hi ** 3) / 12, (h * b ** 3 - hi * bi ** 3) / 12, h / 2, Z, J,
                1 / (2 * Am * t), (2 * h * t) / (b * h - bi * hi), b=b, h=h, t=t, Zy=h * b ** 2 / 4 - hi * bi ** 2 / 4)


@block("structures", "sections", "i_beam", "properties", kind="builder",
       source="Roark 7e Table A.1; torsion open thin-wall sum b t^3/3", tags=("I", "H", "wide flange"))
def i_beam(bf, tf, hw, tw):
    """Symmetric I: flange width bf, flange thickness tf, web clear height hw, web thickness tw."""
    h = hw + 2 * tf
    A = 2 * bf * tf + hw * tw
    Ix = (bf * h ** 3 - (bf - tw) * hw ** 3) / 12
    Iy = 2 * tf * bf ** 3 / 12 + hw * tw ** 3 / 12
    Z = bf * tf * (h - tf) + tw * hw ** 2 / 4
    J = (2 * bf * tf ** 3 + hw * tw ** 3) / 3
    return _sec(A, Ix, Iy, h / 2, Z, J, max(tf, tw) / J, hw * tw / A, h=h, Zy=tf * bf ** 2 / 2 + hw * tw ** 2 / 4)


@block("structures", "sections", "polygon", "properties", kind="builder",
       source="Green's theorem (shoelace) area moments", tags=("drawing", "arbitrary", "CAD"))
def polygon_section(xy):
    """Arbitrary simple polygon section from vertex list [(x,y),...] (drawing/CAD input).
    Returns properties about the centroidal x axis plus centroid. Z by numerical fiber split."""
    P = np.asarray(xy, float)
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    sgn = 1.0 if A > 0 else -1.0
    A = abs(A)
    cx = sgn * ((x + x1) * cr).sum() / (6 * A)
    cy = sgn * ((y + y1) * cr).sum() / (6 * A)
    Ixo = sgn * ((y * y + y * y1 + y1 * y1) * cr).sum() / 12
    Iyo = sgn * ((x * x + x * x1 + x1 * x1) * cr).sum() / 12
    Ix = Ixo - A * cy ** 2
    Iy = Iyo - A * cx ** 2
    c = max(y.max() - cy, cy - y.min())
    # plastic neutral axis: equal-area split, by horizontal strips
    ys = np.linspace(y.min(), y.max(), 2001)
    w = np.array([_chord(P, yy) for yy in ys])
    dA = 0.5 * (w[1:] + w[:-1]) * np.diff(ys)
    cum = np.concatenate([[0], np.cumsum(dA)])
    ypna = np.interp(cum[-1] / 2, cum, ys)
    ym = 0.5 * (ys[1:] + ys[:-1])
    Z = float(np.sum(dA * np.abs(ym - ypna)))
    Zy = _plastic_z(P[:, ::-1].copy()) if len(P) >= 3 else float("nan")
    return _sec(A, Ix, Iy, c, Z, float("nan"), float("nan"), 5 / 6, cx=cx, cy=cy, y_pna=ypna, Zy=Zy)


def _plastic_z(P):
    """Plastic modulus about a horizontal axis by equal-area strips (P is the vertex array)."""
    y = P[:, 1]
    ys = np.linspace(y.min(), y.max(), 2001)
    w = np.array([_chord(P, yy) for yy in ys])
    dA = 0.5 * (w[1:] + w[:-1]) * np.diff(ys)
    cum = np.concatenate([[0], np.cumsum(dA)])
    ypna = np.interp(cum[-1] / 2, cum, ys)
    ym = 0.5 * (ys[1:] + ys[:-1])
    return float(np.sum(dA * np.abs(ym - ypna)))


def _chord(P, y0):
    """Total width of polygon P at height y0 (sum of interior intervals)."""
    xs = []
    n = len(P)
    for i in range(n):
        (xa, ya), (xb, yb) = P[i], P[(i + 1) % n]
        if (ya <= y0 < yb) or (yb <= y0 < ya):
            xs.append(xa + (y0 - ya) * (xb - xa) / (yb - ya))
    xs.sort()
    return sum(xs[i + 1] - xs[i] for i in range(0, len(xs) - 1, 2))


@block("structures", "sections", "general", "plastic", source="Mp = sigma_y Z", tags=("plastic moment",))
def plastic_moment(sy, Z):
    """Fully plastic moment Mp = sy * Z (per unit width for plate_per_width)."""
    return sy * Z


@block("structures", "sections", "general", "shape_factor", source="Z / S",
       tags=("plastic", "reserve"))
def shape_factor(sec):
    """Plastic reserve Z/S (1.5 rectangle, 1.70 circle, ~1.1-1.2 I-beam)."""
    return sec["Z"] / sec["S"]

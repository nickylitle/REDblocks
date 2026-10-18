"""Circular plates, small deflection (Kirchhoff). Roark 7th ed. Table 11.2; Timoshenko & Woinowsky-Krieger.
a = outer radius, t = thickness, D = E t^3 / 12(1-nu^2). All stiffnesses at plate center.
Point-load stresses use Roark's equivalent loaded radius r0' (Roark 11.1, eq. for r0 < 0.5 t)."""
import math

from ...registry import block

SRC = "Roark 7e Table 11.2"


@block("structures", "plates", "general", "rigidity", source="Kirchhoff plate theory")
def flexural_rigidity(E, t, nu):
    """Plate flexural rigidity D = E t^3 / (12 (1 - nu^2))."""
    return E * t ** 3 / (12.0 * (1.0 - nu * nu))


@block("structures", "plates", "general", "point_load_radius", source="Roark 7e section 11.1",
       tags=("tooth contact", "patch"))
def equivalent_load_radius(r0, t):
    """Roark equivalent radius r0' for a load over a small radius r0 (< 0.5 t)."""
    if r0 >= 0.5 * t:
        return r0
    return math.sqrt(1.6 * r0 * r0 + t * t) - 0.675 * t


@block("structures", "plates", "circular_clamped", "center_point", kind="builder",
       source=SRC + " case 16 (fixed edge, r0 -> 0)", tags=("dent", "penetration stiffness"))
def circ_clamped_center_point(mat, a, t, r0=0.0):
    """Clamped circular plate, central point load: k = 16 pi D / a^2.
    Stress at edge 3P/(2 pi t^2); center (with r0') 3P(1+nu)/(2 pi t^2) ln(a/r0')."""
    D = flexural_rigidity(mat.E, t, mat.nu)
    r0e = equivalent_load_radius(max(r0, 1e-9), t)
    s_center = 3 * (1 + mat.nu) / (2 * math.pi * t * t) * math.log(a / r0e)
    s_edge = 3 / (2 * math.pi * t * t)
    smax = max(s_center, s_edge)
    return dict(k=16 * math.pi * D / a ** 2, D=D, stress_per_N=smax, F_y=mat.sy / smax)


@block("structures", "plates", "circular_ss", "center_point", kind="builder",
       source=SRC + " case 16 (simply supported, r0 -> 0)")
def circ_ss_center_point(mat, a, t, r0=0.0):
    """Simply supported circular plate, central point load: k = 16 pi D (1+nu) / ((3+nu) a^2).
    Center stress 3P/(2 pi t^2) [(1+nu) ln(a/r0') + 1]."""
    D = flexural_rigidity(mat.E, t, mat.nu)
    r0e = equivalent_load_radius(max(r0, 1e-9), t)
    s = 3 / (2 * math.pi * t * t) * ((1 + mat.nu) * math.log(a / r0e) + 1)
    return dict(k=16 * math.pi * D * (1 + mat.nu) / ((3 + mat.nu) * a * a), D=D, stress_per_N=s, F_y=mat.sy / s)


@block("structures", "plates", "circular_clamped", "uniform_pressure", kind="builder", source=SRC + " case 10b")
def circ_clamped_pressure(mat, a, t):
    """Clamped circular plate, uniform pressure q: w0 = q a^4/64D, edge stress 3qa^2/(4t^2).
    Returns per unit pressure: w_per_q, stress_per_q, q_y."""
    D = flexural_rigidity(mat.E, t, mat.nu)
    s = 3 * a * a / (4 * t * t)
    return dict(w_per_q=a ** 4 / (64 * D), stress_per_q=s, q_y=mat.sy / s, D=D)


@block("structures", "plates", "circular_ss", "uniform_pressure", kind="builder", source=SRC + " case 10a")
def circ_ss_pressure(mat, a, t):
    """SS circular plate, uniform pressure: w0 = (5+nu) q a^4 / (64 (1+nu) D), center stress 3(3+nu)qa^2/(8t^2)."""
    D = flexural_rigidity(mat.E, t, mat.nu)
    nu = mat.nu
    s = 3 * (3 + nu) * a * a / (8 * t * t)
    return dict(w_per_q=(5 + nu) * a ** 4 / (64 * (1 + nu) * D), stress_per_q=s, q_y=mat.sy / s, D=D)


@block("structures", "plates", "circular_clamped", "large_deflection_pressure",
       source="Roark 7e Table 11.8 case 2 (Way 1934)", tags=("membrane", "large deflection"))
def circ_clamped_pressure_large(mat, a, t, w0):
    """Pressure q for center deflection w0 including membrane stiffening (clamped, edge held):
    q a^4/(E t^4) = 5.33/(1-nu^2) (w/t) + 2.6/(1-nu^2) (w/t)^3. A fit good to w0 ~ t; past that it runs 3-8 %
    too flexible: plate_case(load='pressure_large') and circular_plate_large_deflection use the exact-limit law."""
    r = w0 / t
    return mat.E * t ** 4 / a ** 4 * (5.33 * r + 2.6 * r ** 3) / (1 - mat.nu ** 2)


@block("structures", "plates", "annular_clamped", "free_inner_edge_load", kind="builder",
       source="Roark 7e Table 11.2 case 1L (outer fixed, inner free, line load at inner edge)",
       tags=("hub", "flange"))
def annular_outer_fixed_inner_load(mat, a, b, t):
    """Annular plate, outer edge (radius a) fixed, load P spread on the free inner edge (radius b).
    Stiffness from exact Kirchhoff solution integrated numerically (no table interpolation)."""
    D = flexural_rigidity(mat.E, t, mat.nu)
    w = _annular_inner_line_deflection(a, b, mat.nu, D, fixed_outer=True)
    return dict(k=1.0 / w, D=D)


def _annular_inner_line_deflection(a, b, nu, D, fixed_outer=True, P=1.0):
    """Exact axisymmetric solution w(r) = C1 + C2 r^2 + C3 ln r + C4 r^2 ln r, shear Q = -P/(2 pi r)."""
    import numpy as np
    # Particular: for total load P on inner edge, D * (1/r d/dr (r d/dr (1/r d/dr (r dw/dr)))) = 0 with
    # Q(r) = -D d/dr(nabla^2 w) = P / (2 pi r)  ->  nabla^2 w = -(P/(2 pi D)) ln r + const
    # => w = -(P/(8 pi D)) r^2 (ln r - 1) + A r^2 + B ln r + C
    def basis(r):
        return np.array([r * r, math.log(r), 1.0])
    def slope(r):
        return np.array([2 * r, 1 / r, 0.0])
    def moment_r(r):  # M_r = -D (w'' + nu w'/r)
        return -D * np.array([2 + 2 * nu, -(1 - nu) / r ** 2, 0.0])
    k0 = -P / (8 * math.pi * D)
    wp = lambda r: k0 * r * r * (math.log(r) - 1)
    wp1 = lambda r: k0 * (2 * r * (math.log(r) - 1) + r)
    wp2 = lambda r: k0 * (2 * (math.log(r) - 1) + 3)
    Mp_r = lambda r: -D * (wp2(r) + nu * wp1(r) / r)
    rows, rhs = [], []
    rows.append(basis(a)); rhs.append(-wp(a))                 # w(a) = 0
    if fixed_outer:
        rows.append(slope(a)); rhs.append(-wp1(a))            # w'(a) = 0
    else:
        rows.append(moment_r(a)); rhs.append(-Mp_r(a))
    rows.append(moment_r(b)); rhs.append(-Mp_r(b))            # free inner edge M_r(b) = 0
    c = np.linalg.solve(np.array(rows), np.array(rhs))
    return abs(float(c @ basis(b) + wp(b)))     # magnitude of load-point deflection

"""Rectangular plates, small deflection. a = long side, b = short side (a >= b), t thickness.

Simply supported cases use the exact Navier double series (independent of any table).
Clamped cases use Roark 7th ed. Table 11.4 coefficients (Timoshenko & Woinowsky-Krieger values),
linearly interpolated in a/b. Roark form: w = alpha q b^4 / (E t^3), sigma = beta q b^2 / t^2 (nu = 0.3).
"""
import math
import numpy as np

from ...registry import block
from .circular import flexural_rigidity, equivalent_load_radius

SRC = "Roark 7e Table 11.4"

# case 8a: all edges fixed, uniform load; beta1 = stress at center of long edge, alpha = center deflection
_CL_U_R = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1e9]
_CL_U_BETA = [0.3078, 0.3834, 0.4356, 0.4680, 0.4872, 0.4974, 0.5000]
_CL_U_ALPHA = [0.0138, 0.0188, 0.0226, 0.0251, 0.0267, 0.0277, 0.0284]
# case 8b: all edges fixed, central point load (alpha for w = alpha W b^2 / E t^3)
_CL_P_ALPHA = [0.0611, 0.0706, 0.0755, 0.0777, 0.0786, 0.0788, 0.0791]
# case 1b: simply supported, central point load stress coefficient beta (with r0')
_SS_P_R = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1e9]
_SS_P_BETA = [0.435, 0.650, 0.789, 0.875, 0.927, 0.958, 1.000]


def _interp(r, xs, ys):
    return float(np.interp(min(r, 1e9), xs, ys))


def _navier(a, b, D, load, x0, y0, xe, ye, terms=61):
    """Navier series deflection at (xe, ye). load: ('uniform', q) | ('point', P) | ('patch', P, u, v)."""
    w = 0.0
    for m in range(1, terms + 1, 1):
        for n in range(1, terms + 1, 1):
            if load[0] == "uniform":
                if m % 2 == 0 or n % 2 == 0:
                    continue
                qmn = 16 * load[1] / (math.pi ** 2 * m * n)
            elif load[0] == "point":
                qmn = 4 * load[1] / (a * b) * math.sin(m * math.pi * x0 / a) * math.sin(n * math.pi * y0 / b)
            else:  # uniform patch u x v centered at (x0, y0), total load P
                P, u, v = load[1], load[2], load[3]
                qmn = (16 * P / (math.pi ** 2 * m * n * u * v) * math.sin(m * math.pi * x0 / a)
                       * math.sin(n * math.pi * y0 / b) * math.sin(m * math.pi * u / (2 * a))
                       * math.sin(n * math.pi * v / (2 * b)))
            den = math.pi ** 4 * D * ((m / a) ** 2 + (n / b) ** 2) ** 2
            w += qmn / den * math.sin(m * math.pi * xe / a) * math.sin(n * math.pi * ye / b)
    return w


@block("structures", "plates", "rectangular_ss", "uniform_pressure", kind="builder",
       source="Navier series (exact); cf. " + SRC + " case 1a", tags=("panel", "armor"))
def rect_ss_pressure(mat, a, b, t):
    """SS rectangle under uniform pressure: center deflection per unit pressure and Roark-form alpha."""
    a, b = max(a, b), min(a, b)
    D = flexural_rigidity(mat.E, t, mat.nu)
    w = _navier(a, b, D, ("uniform", 1.0), 0, 0, a / 2, b / 2, 41)
    return dict(w_per_q=w, alpha=w * mat.E * t ** 3 / b ** 4, D=D)


@block("structures", "plates", "rectangular_ss", "center_point", kind="builder",
       source="Navier series (exact); stress " + SRC + " case 1b", tags=("dent", "penetration stiffness"))
def rect_ss_center_point(mat, a, b, t, r0=0.0):
    """SS rectangle, central point load: k at load, and center stress per N with Roark r0' correction."""
    a, b = max(a, b), min(a, b)
    D = flexural_rigidity(mat.E, t, mat.nu)
    w = _navier(a, b, D, ("point", 1.0), a / 2, b / 2, a / 2, b / 2, 81)
    r0e = equivalent_load_radius(max(r0, 1e-9), t)
    beta = _interp(a / b, _SS_P_R, _SS_P_BETA)
    s = 3 / (2 * math.pi * t * t) * ((1 + mat.nu) * math.log(2 * b / (math.pi * r0e)) + beta)
    return dict(k=1 / w, D=D, stress_per_N=s, F_y=mat.sy / s)


@block("structures", "plates", "rectangular_ss", "point_anywhere", kind="builder",
       source="Navier series (exact)", tags=("off-center hit",))
def rect_ss_point_at(mat, a, b, t, x0, y0, patch=(0.0, 0.0)):
    """SS rectangle (0..a, 0..b), load at (x0, y0), optionally over a u x v patch. Stiffness at the load."""
    D = flexural_rigidity(mat.E, t, mat.nu)
    u, v = patch
    if not (0.0 <= x0 <= a and 0.0 <= y0 <= b):
        raise ValueError(f"the load point ({x0:g}, {y0:g}) is off the plate (0..{a:g}, 0..{b:g}): check units and a / b")
    if u > 0 and v > 0 and not (x0 - u / 2 >= -1e-12 and x0 + u / 2 <= a + 1e-12 and y0 - v / 2 >= -1e-12 and
                                y0 + v / 2 <= b + 1e-12):
        raise ValueError(f"the {u:g} x {v:g} patch at ({x0:g}, {y0:g}) runs off the plate (0..{a:g}, 0..{b:g})")
    load = ("patch", 1.0, u, v) if u > 0 and v > 0 else ("point", 1.0)
    w = _navier(a, b, D, load, x0, y0, x0, y0, 81)
    return dict(k=1 / w, D=D)


@block("structures", "plates", "rectangular_clamped", "uniform_pressure", kind="builder",
       source=SRC + " case 8a")
def rect_clamped_pressure(mat, a, b, t):
    """All edges clamped, uniform pressure: w = alpha q b^4/(E t^3), sigma_edge = beta1 q b^2 / t^2."""
    a, b = max(a, b), min(a, b)
    r = a / b
    al, be = _interp(r, _CL_U_R, _CL_U_ALPHA), _interp(r, _CL_U_R, _CL_U_BETA)
    s = be * b * b / t ** 2
    return dict(w_per_q=al * b ** 4 / (mat.E * t ** 3), stress_per_q=s, q_y=mat.sy / s, alpha=al, beta=be)


@block("structures", "plates", "rectangular_clamped", "center_point", kind="builder",
       source=SRC + " case 8b")
def rect_clamped_center_point(mat, a, b, t):
    """All edges clamped, central point load: w = alpha W b^2 / (E t^3)."""
    a, b = max(a, b), min(a, b)
    al = _interp(a / b, _CL_U_R, _CL_P_ALPHA)
    return dict(k=mat.E * t ** 3 / (al * b * b), alpha=al)

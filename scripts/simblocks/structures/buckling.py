"""Elastic stability: columns (Roark 7e Table 15.1, Euler and Johnson) and plates (Table 15.2)."""
import math

from ..registry import block

K_FACTORS = {"pinned-pinned": 1.0, "fixed-free": 2.0, "fixed-pinned": 0.699, "fixed-fixed": 0.5,
             "fixed-guided": 1.0}


@block("structures", "buckling", "column", "axial", source="Euler; Roark 7e Table 15.1",
       tags=("strut", "standoff"))
def euler_load(E, I, L, ends="pinned-pinned"):
    """Euler critical load pi^2 E I / (K L)^2. ends in K_FACTORS."""
    K = K_FACTORS[ends]
    return math.pi ** 2 * E * I / (K * L) ** 2


@block("structures", "buckling", "column", "axial_inelastic", source="J.B. Johnson parabola (Shigley 4-43)",
       tags=("short column",))
def johnson_load(E, sy, A, I, L, ends="pinned-pinned"):
    """Critical load with Johnson parabola below the transition slenderness, Euler above."""
    K = K_FACTORS[ends]
    r = math.sqrt(I / A)
    sl = K * L / r
    sl_t = math.sqrt(2 * math.pi ** 2 * E / sy)
    if sl >= sl_t:
        return math.pi ** 2 * E * A / sl ** 2
    return A * (sy - (sy * sl / (2 * math.pi)) ** 2 / E)


PLATE_K = {"ss_all_compression": 4.0, "clamped_all_compression": 6.97, "one_free_ss_compression": 0.425,
           "one_free_clamped_compression": 1.277, "ss_all_shear": 5.35, "clamped_all_shear": 8.98}


@block("structures", "buckling", "plate", "in_plane", source="Roark 7e Table 15.2 (long plates)",
       tags=("panel", "skin", "web"))
def plate_buckling_stress(E, nu, t, b, case="ss_all_compression"):
    """sigma_cr = k pi^2 E / (12 (1 - nu^2)) (t/b)^2, b = loaded width (short side for long plates)."""
    return PLATE_K[case] * math.pi ** 2 * E / (12 * (1 - nu * nu)) * (t / b) ** 2


@block("structures", "buckling", "thin_cylinder", "axial", source="Roark 7e Table 15.2 case 15 (classical)",
       tags=("tube", "knockdown"))
def cylinder_axial_buckling(E, nu, t, R, knockdown=0.3):
    """Classical axial buckling of a thin cylinder E t / (R sqrt(3(1-nu^2))) times an imperfection knockdown."""
    return knockdown * E * t / (R * math.sqrt(3 * (1 - nu * nu)))

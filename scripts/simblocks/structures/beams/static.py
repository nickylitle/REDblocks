"""Straight beams, Euler-Bernoulli (Roark 7th ed. Table 8.1) plus Timoshenko shear correction
and rigid-plastic collapse loads.

Every builder returns a dict:
  k        point stiffness at the load point (N/m)   [or k_theta N.m/rad for moment loads]
  delta    deflection at the load point per unit load (m/N), = 1/k
  M_max    peak bending moment per unit load (N.m/N), and x_Mmax its location
  F_y      first-yield load = sy * S / M_max  (needs sec + sy)
  F_c      rigid-plastic collapse load (mechanism, needs sec Z + sy)
x is measured from the LEFT / fixed end. a = load position from the left end.
"""
import math

from ...registry import block

SRC = "Roark 7e Table 8.1"


def _out(EI, coef_delta, M_coef, x_M, Fc_coef, sec=None, sy=None, GA=None, shear_coef=0.0, L=None):
    flex = coef_delta / EI
    if GA and shear_coef:
        flex += shear_coef / GA
    d = dict(k=1.0 / flex, delta=flex, M_max=M_coef, x_Mmax=x_M)
    if sec is not None and sy is not None:
        d["F_y"] = sy * sec["S"] / M_coef
        d["F_c"] = Fc_coef * sy * sec["Z"]
        d["Mp"] = sy * sec["Z"]
    return d


def _EI_GA(mat, sec, timoshenko):
    EI = mat.E * sec["Ix"]
    GA = mat.G * sec["A"] * sec["kappa"] if timoshenko else None
    return EI, GA


@block("structures", "beams", "cantilever", "end_point", kind="builder", source=SRC + " case 1a",
       tags=("tip load",))
def cantilever_end_load(mat, sec, L, timoshenko=False):
    """Cantilever, point load at free end. delta = PL^3/3EI (+ PL/kGA)."""
    EI, GA = _EI_GA(mat, sec, timoshenko)
    return _out(EI, L ** 3 / 3, L, 0.0, 1 / L, sec, mat.sy, GA, L)


@block("structures", "beams", "cantilever", "point_at_a", kind="builder", source=SRC + " case 1a (general a)")
def cantilever_point_at(mat, sec, L, a, timoshenko=False):
    """Cantilever (fixed at x=0), point load at distance a from the fixed end; stiffness at the load."""
    EI, GA = _EI_GA(mat, sec, timoshenko)
    return _out(EI, a ** 3 / 3, a, 0.0, 1 / a, sec, mat.sy, GA, a)


@block("structures", "beams", "cantilever", "tip_deflection_point_at_a", source=SRC + " case 1a")
def cantilever_tip_deflection_point_at(P, E, I, L, a):
    """Free-end deflection of a cantilever with load P at a from the fixed end: P a^2 (3L - a) / 6EI."""
    return P * a * a * (3 * L - a) / (6 * E * I)


@block("structures", "beams", "cantilever", "uniform", kind="builder", source=SRC + " case 2a")
def cantilever_uniform(mat, sec, L):
    """Cantilever, uniform load w total W = wL. Returns per unit TOTAL load W. tip delta = WL^3/8EI."""
    EI = mat.E * sec["Ix"]
    return _out(EI, L ** 3 / 8, L / 2, 0.0, 2 / L, sec, mat.sy)


@block("structures", "beams", "cantilever", "end_moment", kind="builder", source=SRC + " case 3a")
def cantilever_end_moment(mat, sec, L):
    """Cantilever with end couple: k_theta = EI/L, tip delta = ML^2/2EI."""
    EI = mat.E * sec["Ix"]
    return dict(k_theta=EI / L, tip_delta_per_M=L * L / (2 * EI), M_y=mat.sy * sec["S"], Mp=mat.sy * sec["Z"])


@block("structures", "beams", "simply_supported", "center_point", kind="builder", source=SRC + " case 1e")
def simply_supported_center(mat, sec, L, timoshenko=False):
    """Simply supported, central point load: delta = PL^3/48EI, Mmax = PL/4, collapse 4Mp/L."""
    EI, GA = _EI_GA(mat, sec, timoshenko)
    return _out(EI, L ** 3 / 48, L / 4, L / 2, 4 / L, sec, mat.sy, GA, L / 4)


@block("structures", "beams", "simply_supported", "point_at_a", kind="builder", source=SRC + " case 1e")
def simply_supported_point_at(mat, sec, L, a):
    """Simply supported, load at a: delta_load = P a^2 b^2 / (3 E I L), M = P a b / L, collapse Mp L/(a b)."""
    b = L - a
    EI = mat.E * sec["Ix"]
    return _out(EI, a * a * b * b / (3 * L), a * b / L, a, L / (a * b), sec, mat.sy)


@block("structures", "beams", "simply_supported", "uniform", kind="builder", source=SRC + " case 2e")
def simply_supported_uniform(mat, sec, L):
    """Simply supported, uniform load (per total W): center delta 5WL^3/384EI, M = WL/8, collapse 8Mp/L."""
    EI = mat.E * sec["Ix"]
    return _out(EI, 5 * L ** 3 / 384, L / 8, L / 2, 8 / L, sec, mat.sy)


@block("structures", "beams", "fixed_fixed", "center_point", kind="builder", source=SRC + " case 1d")
def fixed_fixed_center(mat, sec, L, timoshenko=False):
    """Both ends fixed, central point load: delta = PL^3/192EI, Mmax = PL/8 (ends and center), collapse 8Mp/L."""
    EI, GA = _EI_GA(mat, sec, timoshenko)
    return _out(EI, L ** 3 / 192, L / 8, 0.0, 8 / L, sec, mat.sy, GA, L / 4)


@block("structures", "beams", "fixed_fixed", "uniform", kind="builder", source=SRC + " case 2d")
def fixed_fixed_uniform(mat, sec, L):
    """Both ends fixed, uniform load (per total W): delta WL^3/384EI, M_end = WL/12, collapse 16Mp/L."""
    EI = mat.E * sec["Ix"]
    return _out(EI, L ** 3 / 384, L / 12, 0.0, 16 / L, sec, mat.sy)


@block("structures", "beams", "fixed_guided", "end_point", kind="builder",
       source=SRC + " case 1b (left fixed, right guided)", tags=("sliding clamp", "series cantilevers"))
def fixed_guided_end_load(mat, sec, L, timoshenko=False):
    """Fixed at one end, guided (no rotation, free translation) at the other, load at the guided end:
    delta = PL^3/12EI, M = PL/2 at both ends, collapse 2Mp/L."""
    EI, GA = _EI_GA(mat, sec, timoshenko)
    return _out(EI, L ** 3 / 12, L / 2, 0.0, 2 / L, sec, mat.sy, GA, L)


@block("structures", "beams", "propped_cantilever", "center_point", kind="builder", source=SRC + " case 1c")
def propped_center(mat, sec, L):
    """Fixed at x=0, simply supported at x=L, central load: delta at load = 7PL^3/768EI,
    M_fixed = 3PL/16, collapse 6Mp/L."""
    EI = mat.E * sec["Ix"]
    return _out(EI, 7 * L ** 3 / 768, 3 * L / 16, 0.0, 6 / L, sec, mat.sy)


@block("structures", "beams", "propped_cantilever", "uniform", kind="builder", source=SRC + " case 2c")
def propped_uniform(mat, sec, L):
    """Fixed-pinned, uniform load (per total W): max delta WL^3/184.6EI, M_fixed = WL/8,
    collapse 11.66 Mp/L (hinge at 0.414 L)."""
    EI = mat.E * sec["Ix"]
    return _out(EI, L ** 3 / 184.6, L / 8, 0.0, 11.657 / L, sec, mat.sy)


@block("structures", "beams", "series", "springs", source="compliance addition",
       tags=("blade + chassis", "load path"))
def springs_in_series(*k):
    """Combined stiffness of springs in series 1/k = sum 1/k_i (e.g. blade cantilever + chassis cantilever)."""
    return 1.0 / sum(1.0 / x for x in k)


@block("structures", "beams", "parallel", "springs", source="stiffness addition")
def springs_in_parallel(*k):
    """Combined stiffness of springs in parallel k = sum k_i."""
    return float(sum(k))


@block("structures", "beams", "dispersion", "effective_width", source="45 deg load spread (St-Venant)",
       tags=("plate cantilever", "root width"))
def effective_width_45(load_width, distance, plate_width=math.inf):
    """Effective resisting width at distance from a concentrated load: load_width + 2*distance, capped."""
    return min(load_width + 2.0 * distance, plate_width)

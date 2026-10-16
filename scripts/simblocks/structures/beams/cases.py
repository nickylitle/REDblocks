"""Generic straight-beam block: one entry point for the Roark Table 8.1 support and load cases.

beam_case dispatches to the specific builders in static.py (cantilever_end_load, simply_supported_center, ...),
which stay importable for hand-written code. It takes plain numbers (E and I, or EI) instead of material and
section objects so a system file can call it directly from derived:.
"""
import math
from types import SimpleNamespace

from ...registry import block
from . import static as S

_SUPPORTS = {
    "cantilever": "cantilever", "simply_supported": "simply_supported", "ss": "simply_supported",
    "pinned_pinned": "simply_supported", "fixed_fixed": "fixed_fixed", "propped": "propped",
    "propped_cantilever": "propped", "fixed_pinned": "propped", "fixed_guided": "fixed_guided",
}
_LOADS = {"point": "point", "center": "point", "end": "point", "uniform": "uniform", "end_moment": "end_moment",
          "moment": "end_moment"}
_CASES = {
    "cantilever": ("point", "uniform", "end_moment"),
    "simply_supported": ("point", "uniform"),
    "fixed_fixed": ("point", "uniform"),
    "propped": ("point", "uniform"),
    "fixed_guided": ("point",),
}


def _close(x, y, L):
    return abs(x - y) <= 1e-9 * L


@block("structures", "beams", "general", "any", kind="builder",
       source="Roark 7e Table 8.1 cases 1a-1e, 2a-2e, 3a; rigid-plastic collapse by the mechanism method",
       tags=("cantilever", "simply supported", "fixed", "propped", "guided", "stiffness", "collapse", "beam"))
def beam_case(support, load, L, E=None, I=None, EI=None, a=None, Mp=None, My=None, GA=None):
    """Straight beam stiffness, moment and collapse load for a support and load case.

    support    cantilever (fixed at x=0, free at x=L), simply_supported, fixed_fixed,
               propped (fixed at x=0, pinned at x=L), fixed_guided (fixed at x=0, guided at x=L: no rotation,
               free translation)
    load       point       point load at a, measured from x=0. Default a: L for cantilever and fixed_guided,
                           L/2 otherwise. Any a for cantilever and simply_supported; fixed_fixed and propped
                           take only the centre (a = L/2); fixed_guided only the guided end (a = L).
               uniform     uniform load, all results per unit TOTAL load W = w L
               end_moment  couple at the free end (cantilever only)
    L          span (m)
    E, I       modulus (Pa) and second moment of area (m^4), or EI (N m^2) directly
    a          load position for a point load (m)
    Mp         fully plastic moment (N m), adds the rigid-plastic collapse load F_c
    My         first-yield moment sy S (N m), adds the first-yield load F_y
    GA         shear rigidity kappa G A (N), adds Timoshenko shear flexibility (point loads on a cantilever end,
               simply supported centre, fixed_fixed centre and fixed_guided end only)

    Returns dict(k, delta, M_max, x_Mmax, plus delta_tip, tip_delta_per_M, F_c, Mp, F_y where they apply):
      k        stiffness at the load point (N/m); for end_moment the rotational stiffness k_theta (N m/rad)
      delta    deflection at the load per unit load (m/N), = 1/k; for end_moment the rotation per unit moment
      M_max    peak bending moment per unit load (N m/N), x_Mmax its position from x=0
      delta_tip  free-end deflection per unit load (cantilever point load only)
      tip_delta_per_M  free-end deflection per unit moment (end_moment only)
      F_c, Mp  collapse load (N; collapse moment for end_moment) when Mp is given
      F_y      first-yield load (N) when My is given
    """
    sup = _SUPPORTS.get(str(support).strip().lower())
    ld = _LOADS.get(str(load).strip().lower())
    if sup is None:
        raise ValueError(f"beam_case: unknown support {support!r}; use one of "
                         "cantilever, simply_supported, fixed_fixed, propped, fixed_guided")
    if ld is None or ld not in _CASES[sup]:
        raise ValueError(f"beam_case: no '{load}' case for support '{sup}'; it takes: {', '.join(_CASES[sup])}")
    if EI is None:
        if E is None or I is None:
            raise ValueError("beam_case needs E and I, or EI")
        E_, I_ = E, I
    else:
        E_, I_ = EI, 1.0
    nan = math.nan
    plastic = Mp is not None or My is not None
    mat = SimpleNamespace(E=E_, G=GA if GA is not None else nan, sy=1.0 if plastic else None)
    sec = {"Ix": I_, "A": 1.0, "kappa": 1.0, "Z": Mp if Mp is not None else nan, "S": My if My is not None else nan}
    tim = GA is not None

    def no_shear(case):
        if tim:
            raise ValueError(f"beam_case: shear correction (GA) is not available for {case}")

    if sup == "cantilever":
        if ld == "end_moment":
            no_shear("an end moment")
            EIv = E_ * I_
            out = dict(k=EIv / L, delta=L / EIv, M_max=1.0, x_Mmax=0.0, tip_delta_per_M=L * L / (2 * EIv))
            if Mp is not None:
                out["F_c"] = Mp; out["Mp"] = Mp
            if My is not None:
                out["F_y"] = My
            return out
        if ld == "uniform":
            no_shear("a uniform load")
            out = S.cantilever_uniform(mat, sec, L)
        else:
            aa = L if a is None else a
            if _close(aa, L, L):
                out = S.cantilever_end_load(mat, sec, L, timoshenko=tim)
            else:
                no_shear("a cantilever point load away from the free end")
                out = S.cantilever_point_at(mat, sec, L, aa)
            out["delta_tip"] = S.cantilever_tip_deflection_point_at(1.0, E_, I_, L, aa)
    elif sup == "simply_supported":
        if ld == "uniform":
            no_shear("a uniform load")
            out = S.simply_supported_uniform(mat, sec, L)
        else:
            aa = L / 2 if a is None else a
            if _close(aa, L / 2, L):
                out = S.simply_supported_center(mat, sec, L, timoshenko=tim)
            else:
                no_shear("an off-centre point load")
                out = S.simply_supported_point_at(mat, sec, L, aa)
    elif sup in ("fixed_fixed", "propped"):
        if ld == "uniform":
            no_shear("a uniform load")
            out = (S.fixed_fixed_uniform if sup == "fixed_fixed" else S.propped_uniform)(mat, sec, L)
        else:
            if a is not None and not _close(a, L / 2, L):
                raise ValueError(f"beam_case: {sup} point load is tabulated at the centre only (a = L/2)")
            if sup == "fixed_fixed":
                out = S.fixed_fixed_center(mat, sec, L, timoshenko=tim)
            else:
                no_shear("a propped cantilever")
                out = S.propped_center(mat, sec, L)
    else:  # fixed_guided
        if a is not None and not _close(a, L, L):
            raise ValueError("beam_case: fixed_guided point load is at the guided end only (a = L)")
        out = S.fixed_guided_end_load(mat, sec, L, timoshenko=tim)
    return {k: v for k, v in out.items() if not (isinstance(v, float) and math.isnan(v))}


@block("structures", "beams", "series", "combination", source="compliance addition (series), stiffness addition "
       "(parallel)", tags=("series", "parallel", "load path", "blade + chassis"))
def spring_combination(k, arrangement="series"):
    """Combined stiffness of a list of springs.
    k            list of stiffnesses (N/m or N m/rad), e.g. [$blade.k, $mount.k]
    arrangement  series (1/k = sum 1/k_i, loads pass through each spring in turn) or
                 parallel (k = sum k_i, springs share the deflection)
    Returns the combined stiffness."""
    ks = [float(x) for x in (k if isinstance(k, (list, tuple)) else [k])]
    arr = str(arrangement).strip().lower()
    if arr == "series":
        return S.springs_in_series(*ks)
    if arr == "parallel":
        return S.springs_in_parallel(*ks)
    raise ValueError("spring_combination: arrangement is 'series' or 'parallel'")

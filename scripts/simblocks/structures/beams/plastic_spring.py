"""Elastic-plastic force-deflection springs for whole beams, built from the material curve.
Collapse by a plastic-hinge mechanism; the hinge moment-rotation law comes from fibre integration of the
section with the real stress-strain curve (moment_curvature_polygon -> hinge_table)."""
import math
import numpy as np

from ...registry import block

_SUPPORT = {
    # single-hinge mechanisms only (exact mapping of one hinge law to the beam):
    # elastic k = factor E I / L^3; F = Pc M / L; hinge rotation theta = th delta / L
    "ss_center": dict(k=48.0, Pc=4.0, th=4.0, note="simply supported, centre load: 1 hinge at mid-span"),
    "cantilever_end": dict(k=3.0, Pc=1.0, th=1.0, note="cantilever, end load: hinge at the root"),
}


def _section_polygon(section):
    if "circle" in section:
        d = float(section["circle"]); th = np.linspace(0, 2 * math.pi, 181)[:-1]
        return np.c_[0.5 * d * np.cos(th), 0.5 * d * np.sin(th)]
    if "rect" in section:
        b, h = (float(x) for x in section["rect"])
        return np.array([[-b / 2, -h / 2], [b / 2, -h / 2], [b / 2, h / 2], [-b / 2, h / 2]])
    if "polygon" in section:
        return np.asarray(section["polygon"], float)
    if "I" in section:                               # given properties: rectangle with the same I and depth
        if section.get("c") is None and section.get("h") is None:
            raise ValueError("a plastic beam spring needs c (distance to the longest fibre) or h with a given I")
        I = float(section["I"]); c = float(section["c"]) if section.get("c") is not None else float(section["h"]) / 2
        h = 2 * c; b = 12 * I / h ** 3
        return np.array([[-b / 2, -h / 2], [b / 2, -h / 2], [b / 2, h / 2], [-b / 2, h / 2]])
    raise ValueError("section must be {circle: d}, {rect: [b, h]}, {polygon: [[x, y], ...]} or {I: .., c: .., Z: ..}")


@block("structures", "beams", "any_section", "plastic_spring", kind="builder",
       source="Roark 7e Table 8.1 elastic stiffness + plastic-hinge mechanism with fibre-integrated M(kappa)",
       tags=("shaft", "plastic bending", "collapse", "elastoplastic spring"))
def beam_plastic_spring(material, section, L, support="ss_center", hinge_len=None, kappa_max_x_depth=0.3):
    """Force-deflection spring (k, acc_tab, F_tab) of a whole beam at its load point: elastic k from the support
    case, plastic part from the hinge mechanism. material = catalog text; section = {circle: d} | {rect: [b, h]}.
    hinge_len defaults to the section depth. Returns dict(k, acc_tab, F_tab, Mp, M_y, F_collapse, source)."""
    from ...materials import get, load_curve
    from ...materials.curves import resolve_material
    from ..sections.moment_curvature import moment_curvature_polygon, hinge_table
    from ..sections.properties import polygon_section
    from ...constitutive.plasticity import hinge_to_spring_table
    sp = _SUPPORT[support]
    key = resolve_material(material); mat = get(key); curve = load_curve(key)
    xy = _section_polygon(section)
    depth = float(xy[:, 1].max() - xy[:, 1].min())
    if "circle" in section:
        I = math.pi * float(section["circle"]) ** 4 / 64.0
    elif "rect" in section:
        b_, h_ = (float(x) for x in section["rect"]); I = b_ * h_ ** 3 / 12.0
    elif "I" in section:
        I = float(section["I"])
    else:
        I = polygon_section(xy)["Ix"]
    k = sp["k"] * mat.E * I / L ** 3
    mk = moment_curvature_polygon(curve, xy, kappa_max=kappa_max_x_depth / depth)
    k_th, thp, M = hinge_to_spring_table(hinge_table(mk, hinge_len or depth))
    if "I" in section:                               # rescale the plastic reserve to the given Z
        c_ = float(section["c"]) if section.get("c") is not None else float(section["h"]) / 2; S_ = I / c_
        Z_rect = 1.5 * S_                            # plastic modulus of the equivalent rectangle
        Z_ = float(section.get("Z") or S_)
        My = mat.sy * S_
        M = np.where(M > My, My + (M - My) * max(Z_ - S_, 0.0) / (Z_rect - S_), M)
    acc = thp * L / sp["th"]
    F = M * sp["Pc"] / L
    return dict(k=float(k), acc_tab=np.ascontiguousarray(acc, float), F_tab=np.ascontiguousarray(F, float),
                M_y=float(M[0]), Mp=float(M.max()), F_collapse=float(F.max()),
                source=f"{support}: {sp['note']}; curve: {curve.source[:80]}")

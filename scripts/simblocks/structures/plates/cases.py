"""Generic plate blocks: elastic plate cases (Roark Table 11.2 / 11.4, Navier series) and rigid-plastic collapse
(yield-line theory). They dispatch to the specific builders in circular.py, rectangular.py and yield_line.py,
which stay importable for hand-written code, and take a material name or plain numbers."""
from ...registry import block
from ...materials.lookup import mat_props, need
from . import circular as C, rectangular as R, yield_line as Y

_SHAPES = {"circular": "circular", "circ": "circular", "circle": "circular",
           "rectangular": "rectangular", "rect": "rectangular", "rectangle": "rectangular",
           "annular": "annular", "annulus": "annular"}
_EDGES = {"clamped": "clamped", "fixed": "clamped", "ss": "ss", "simply_supported": "ss", "pinned": "ss",
          "outer_clamped": "clamped"}
_PLATE_CASES = {
    ("circular", "clamped"): ("point", "pressure", "pressure_large"),
    ("circular", "ss"): ("point", "pressure"),
    ("rectangular", "ss"): ("point", "point_at", "pressure"),
    ("rectangular", "clamped"): ("point", "pressure"),
    ("annular", "clamped"): ("inner_edge",),
}


@block("structures", "plates", "general", "any", kind="builder",
       source="Roark 7e Tables 11.2 (cases 1L, 10a, 10b, 16), 11.4 (cases 1b, 8a, 8b), 11.8 case 2; "
              "Navier double series for simply supported rectangles",
       tags=("plate", "dent", "panel", "armor", "stiffness", "circular", "rectangular", "annular"))
def plate_case(shape, edges, load, a, t, b=None, material=None, E=None, nu=None, sy=None, r0=0.0,
               x0=None, y0=None, patch_u=0.0, patch_v=0.0, w0=None):
    """Small-deflection plate stiffness and stress for a shape, edge condition and load.

    shape, edges, load:
      circular    clamped  point           central point load: k, D, stress_per_N, F_y (r0 = loaded radius)
      circular    clamped  pressure        uniform pressure: w_per_q, stress_per_q, q_y, D
      circular    clamped  pressure_large  pressure q that gives centre deflection w0 with membrane stiffening
                                           (edge held): dict(q)
      circular    ss       point           central point load: k, D, stress_per_N, F_y
      circular    ss       pressure        uniform pressure: w_per_q, stress_per_q, q_y, D
      rectangular ss       point           central point load: k, D, stress_per_N, F_y
      rectangular ss       point_at        load at (x0, y0) on 0..a x 0..b, optional patch_u x patch_v: k, D
      rectangular ss       pressure        uniform pressure: w_per_q, alpha, D
      rectangular clamped  point           central point load: k, alpha
      rectangular clamped  pressure        uniform pressure: w_per_q, stress_per_q, q_y, alpha, beta
      annular     clamped  inner_edge      outer edge (radius a) clamped, load spread on the free inner edge
                                           (radius b): k, D
    a          circular: outer radius; rectangular: one side (the longer side is used as a); annular: outer radius
    b          rectangular: other side; annular: inner radius
    t          thickness (m)
    material   catalog key or loose text ('7075', '4130 45 HRC'); E, nu, sy override it or replace it
    r0         loaded radius for point-load stresses (Roark equivalent radius applied)
    Returns the case's dict as listed: point loads k (N/m), D, stress_per_N, F_y; pressure w_per_q, stress_per_q,
    q_y; pressure_large q (Pa). Stresses are per unit load or pressure; F_y and q_y are first-yield loads.
    """
    sh = _SHAPES.get(str(shape).strip().lower())
    ed = _EDGES.get(str(edges).strip().lower())
    ld = str(load).strip().lower()
    if sh is None:
        raise ValueError(f"plate_case: unknown shape {shape!r}; use circular, rectangular or annular")
    if ed is None:
        raise ValueError(f"plate_case: unknown edges {edges!r}; use clamped or ss")
    ok = _PLATE_CASES.get((sh, ed), ())
    if ld not in ok:
        raise ValueError(f"plate_case: no '{load}' case for a {sh} plate with {ed} edges"
                         + (f"; it takes: {', '.join(ok)}" if ok else "; annular plates take edges: clamped"))
    m = mat_props(material, E=E, nu=nu, sy=sy)
    need("E (or material)", m.E, "plate_case")
    need("nu (or material)", m.nu, "plate_case")
    if sh == "circular":
        if ld == "point":
            return (C.circ_clamped_center_point if ed == "clamped" else C.circ_ss_center_point)(m, a, t, r0)
        if ld == "pressure":
            return (C.circ_clamped_pressure if ed == "clamped" else C.circ_ss_pressure)(m, a, t)
        res = dict(q=C.circ_clamped_pressure_large(m, a, t, need("w0", w0, "plate_case pressure_large")))
        return res
    if sh == "annular":
        return C.annular_outer_fixed_inner_load(m, a, need("b (inner radius)", b, "plate_case annular"), t)
    need("b", b, "plate_case rectangular")
    if ed == "ss":
        if ld == "point":
            return R.rect_ss_center_point(m, a, b, t, r0)
        if ld == "point_at":
            return R.rect_ss_point_at(m, a, b, t, need("x0", x0, "plate_case point_at"),
                                      need("y0", y0, "plate_case point_at"), patch=(patch_u, patch_v))
        return R.rect_ss_pressure(m, a, b, t)
    if ld == "point":
        return R.rect_clamped_center_point(m, a, b, t)
    return R.rect_clamped_pressure(m, a, b, t)


_COLLAPSE = ("point_interior", "point_free_edge", "circular_pressure", "rect_pressure", "three_edge")


@block("structures", "plates", "any_shape", "collapse", source=Y.SRC + "; three_edge approximate fold mechanism",
       tags=("collapse", "yield line", "puncture precursor", "plate edge", "hinged plate"))
def plate_collapse(case, mp=None, sy=None, t=None, a=None, b=None, clamped=False, b_eff=None, L_slant=None):
    """Rigid-plastic collapse capacity of a plate by yield-line theory (upper bound).

    case
      point_interior     point load at an interior point of any plate: 2 pi m_p (simply supported edges),
                         4 pi m_p (clamped). Returns a force (N).
      point_free_edge    point load at a free edge, half fan: pi m_p. Force (N).
      circular_pressure  uniform pressure on a circular plate of radius a: 6 m_p/a^2 (SS), 12 m_p/a^2 (clamped).
                         Pressure (Pa).
      rect_pressure      uniform pressure on an a x b rectangle: 24 m_p / (b^2 (sqrt(3 + beta^2) - beta)^2),
                         beta = b/a (SS), doubled when clamped. Pressure (Pa).
      three_edge         face clamped on three edges, loaded near the free lip: m_p (b_eff + 2 L_slant).
                         Collapse moment about the root (N m).
    mp         plastic moment per unit width (N m/m); or give sy and t to use m_p = sy t^2 / 4
    clamped    true for clamped edges (point_interior, circular_pressure, rect_pressure)
    Returns dict(capacity, mp).
    """
    c = str(case).strip().lower()
    if c not in _COLLAPSE:
        raise ValueError(f"plate_collapse: unknown case {case!r}; use one of {', '.join(_COLLAPSE)}")
    if mp is None:
        if sy is None or t is None:
            raise ValueError("plate_collapse needs mp, or sy and t")
        mp = Y.plate_mp(sy, t)
    cl = bool(clamped)
    if c == "point_interior":
        cap = Y.collapse_point_interior(mp, cl)
    elif c == "point_free_edge":
        cap = Y.collapse_point_free_edge(mp)
    elif c == "circular_pressure":
        cap = Y.collapse_circular_pressure(mp, need("a", a, "plate_collapse circular_pressure"), cl)
    elif c == "rect_pressure":
        cap = Y.collapse_rect_pressure(mp, need("a", a, "plate_collapse rect_pressure"),
                                       need("b", b, "plate_collapse rect_pressure"), cl)
    else:
        cap = Y.collapse_three_edge_strip(mp, need("b_eff", b_eff, "plate_collapse three_edge"),
                                          need("L_slant", L_slant, "plate_collapse three_edge"))
    return dict(capacity=cap, mp=mp)

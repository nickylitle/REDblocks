"""Laminate plate under a central patch load: stiffness and first-ply-failure load in one builder."""
import math

from ...registry import block


@block("structures", "plates", "laminate_rect_ss", "center_patch_fpf", kind="builder",
       source="CLT laminate (Tsai-Wu FPF) + Navier SS plate stiffness + Roark 11.4 centre moment with r0",
       tags=("carbon plate", "motor mount", "first-ply failure"))
def laminate_plate_point_load(layup, a, b, patch, support="ss", equal_moments=True):
    """Rectangular laminate plate, simply supported, square patch (side `patch`) at the centre.
    Returns dict(k, F_fpf, t, E_flex, ply, angle). Mx = My at the centre when equal_moments (near-square)."""
    from ...materials.cards import laminate_card
    from ...materials import clt
    from .rectangular import rect_ss_point_at, rect_ss_center_point
    if support != "ss":
        raise ValueError("only simply supported edges for now")
    card = laminate_card(layup); abd = card.abd; iso = card.as_isotropic("flexural"); t = abd["h"]
    k = rect_ss_point_at(iso, a, b, t, a / 2, b / 2, patch=(patch, patch))["k"]
    r0 = patch / math.sqrt(math.pi)
    s_per_N = rect_ss_center_point(iso, a, b, t, r0=r0)["stress_per_N"]
    m = s_per_N * t * t / 6.0
    R, ply, ang = clt.first_ply_failure(abd, M=(m, m if equal_moments else 0.0, 0.0))
    return dict(k=float(k), F_fpf=float(R), t=float(t), E_flex=float(iso.E), ply=int(ply), angle=float(ang))

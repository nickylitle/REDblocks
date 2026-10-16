"""Rigid-plastic beams with membrane (string) action at finite deflection: the load keeps rising after the
collapse load because axially restrained hinges stretch. Jones, Structural Impact (2nd ed. 2012) eqs. 1.5,
1.36, 1.37 for M0 and Pc (VERIFIED); the finite-deflection laws (Jones sec. 7.2, eq. 7.39) were re-derived by
upper bound with the hinge yield condition m + n^2 = 1 and checked against the pure-string limit of Shi,
Appl. Sci. 13 (2023) 9051 eq. 26 (PARTIAL: Jones ch. 7 itself not opened).
Valid only with axially IMMOVABLE supports; a free or sliding edge gets no membrane gain."""
from ...registry import block

SRC = ("Jones, Structural Impact 2nd ed. eqs. 1.5, 1.36, 1.37 (Pc, verified); finite-deflection law re-derived "
       "(m + n^2 = 1 upper bound), string limit checked vs Shi 2023 eq. 26. PARTIAL")


@block("structures", "beams", "restrained", "point_load_membrane", source=SRC,
       tags=("membrane", "large deflection", "rigid-plastic", "string action", "dent"))
def beam_membrane_point_load(W, H, B, L_half, sigma0, support="clamped"):
    """Central point load P on a rigid-plastic rectangular beam (depth H, breadth B, span 2 L_half) at midspan
    deflection W, axially restrained ends. Returns (P, Pc, P/Pc).
    clamped:  Pc = 4 M0 / L,  P/Pc = 1 + (W/H)^2 for W/H <= 1,   2 W/H beyond.
    pinned:   Pc = 2 M0 / L,  P/Pc = 1 + 4 (W/H)^2 for W/H <= 1/2, 4 W/H beyond.
    Both reach the same string asymptote P = 2 N0 W / L (N0 = sigma0 B H)."""
    M0 = sigma0 * B * H * H / 4.0
    x = W / H
    if support == "clamped":
        Pc = 4.0 * M0 / L_half
        r = 1.0 + x * x if x <= 1.0 else 2.0 * x
    elif support == "pinned":
        Pc = 2.0 * M0 / L_half
        r = 1.0 + 4.0 * x * x if x <= 0.5 else 4.0 * x
    else:
        raise ValueError(support)
    return Pc * r, Pc, r

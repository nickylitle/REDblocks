"""Curved beams and rings.
Winkler-Bach curved beam (Shigley 10e sec. 3-18; Roark 7e sec. 9.1).
Thin rings solved by numerical Castigliano (exact for Euler-Bernoulli rings): no table coefficients needed,
reproduces Roark 7e Table 9.2 values."""
import math
import numpy as np

from ..registry import block


@block("structures", "curved", "rectangle", "bending", source="Winkler-Bach; Shigley 10e eq. 3-65, Table 3-4",
       tags=("hook", "C-frame", "curved arm"))
def curved_beam_rect(M, b, h, r_inner, N=0.0):
    """Stresses at inner and outer fibers of a curved rectangular beam (b wide, h deep, inner radius r_inner)
    under moment M (positive closes the curvature) and normal force N. Returns (sigma_i, sigma_o, e)."""
    ri, ro = r_inner, r_inner + h
    rc = ri + h / 2
    rn = h / math.log(ro / ri)
    e = rc - rn
    A = b * h
    si = M * (rn - ri) / (A * e * ri) + N / A
    so = M * (rn - ro) / (A * e * ro) + N / A
    return si, so, e


@block("structures", "curved", "thin_ring", "diametral_point_loads", kind="builder",
       source="Castigliano + Mohr curvature integrals (numerical, exact); cf. Roark 7e Table 9.2 case 1",
       tags=("ring", "hoop", "drum", "bearing race"))
def ring_diametral_load(P, R, EI, n=20001):
    """Thin ring radius R squeezed by equal and opposite P along a diameter (Euler-Bernoulli, inextensible).
    Quarter ring, psi from the side section (psi=0) to the load point (psi=pi/2).
    M(psi) = M0 - (P/2) R (1 - cos psi); zero rotation at the side section fixes M0.
    Returns dD_load (shortening along the load line), dD_perp (growth across), M_load, M_side (both signed),
    k_diametral = P / dD_load."""
    psi = np.linspace(0.0, math.pi / 2, n)
    Ml = -(P / 2) * R * (1 - np.cos(psi))
    M0 = -np.trapezoid(Ml, psi) / (math.pi / 2)
    M = M0 + Ml
    dD_load = 2 * abs(np.trapezoid(M * R * R * np.cos(psi), psi)) / EI
    dD_perp = 2 * abs(np.trapezoid(M * R * R * np.sin(psi), psi)) / EI
    return dict(dD_load=dD_load, dD_perp=dD_perp, M_load=float(M[-1]), M_side=float(M[0]),
                M_max=float(np.max(np.abs(M))), k_diametral=P / dD_load)

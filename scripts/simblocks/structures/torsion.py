"""Torsion of bars and shafts (Roark 7th ed. Table 10.1, Saint-Venant) plus keys and combined loading."""
import math

from ..registry import block


@block("structures", "torsion", "any_section", "twist", source="Roark 7e eq. 10.1-1",
       tags=("shaft", "torsional stiffness"))
def torsion_stiffness(G, J, L):
    """Torsional stiffness k_theta = G J / L (N.m/rad). Use J from a sections builder."""
    return G * J / L


@block("structures", "torsion", "any_section", "stress", source="Roark 7e Table 10.1")
def torsion_stress(T, sec):
    """Max shear stress tau = tau_coef * T, using the section's Roark coefficient."""
    return sec["tau_coef"] * T


@block("structures", "torsion", "circle", "plastic", source="fully plastic solid shaft")
def torsion_plastic_solid(tau_y, d):
    """Fully plastic torque of a solid round shaft: T_p = (2/3) pi tau_y r^3 (4/3 of first yield)."""
    r = d / 2
    return 2.0 / 3.0 * math.pi * tau_y * r ** 3


@block("structures", "torsion", "circle", "combined", source="von Mises, Shigley 5-15",
       tags=("bending + torsion", "rotor shaft"))
def shaft_von_mises(M, T, d, N=0.0):
    """Von Mises stress at the surface of a solid round shaft under bending M, torque T, axial N."""
    s = 32 * M / (math.pi * d ** 3) + 4 * N / (math.pi * d * d)
    t = 16 * T / (math.pi * d ** 3)
    return math.sqrt(s * s + 3 * t * t)


@block("structures", "torsion", "keyed_shaft", "shear", source="Shigley 7-8")
def key_shear(T, d, w, L):
    """Square/rectangular key shear stress tau = 2T / (d w L)."""
    return 2 * T / (d * w * L)


@block("structures", "torsion", "keyed_shaft", "bearing", source="Shigley 7-8")
def key_bearing(T, d, h, L):
    """Key bearing stress on half key height: sigma = 4T / (d h L)."""
    return 4 * T / (d * h * L)

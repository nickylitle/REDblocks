"""Elastic contact (Hertz) and contact-duration estimates. Roark 7th ed. Table 14.1; Johnson, Contact Mechanics."""
import math

from ..registry import block
from .._jit import kernel


@block("contact", "hertz", "general", "effective", source="Johnson, Contact Mechanics ch. 4")
def effective_modulus(E1, nu1, E2, nu2):
    """E* = 1 / ((1-nu1^2)/E1 + (1-nu2^2)/E2)."""
    return 1.0 / ((1 - nu1 * nu1) / E1 + (1 - nu2 * nu2) / E2)


@block("contact", "hertz", "general", "effective", source="Johnson, Contact Mechanics ch. 4")
def effective_radius(R1, R2=math.inf):
    """R* = 1/(1/R1 + 1/R2). Flat surface: R2 = inf. Concave: negative radius."""
    inv = 1.0 / R1 + (0.0 if math.isinf(R2) else 1.0 / R2)
    return 1.0 / inv


@block("contact", "hertz", "sphere", "normal_force", kind="kernel", source="Roark 7e Table 14.1 case 1",
       tags=("tooth tip", "ball", "nonlinear spring"))
@kernel
def hertz_sphere_force(delta, Estar, Rstar):
    """Sphere-on-sphere/flat normal force F = (4/3) E* sqrt(R*) delta^1.5 (0 if delta <= 0)."""
    if delta <= 0.0:
        return 0.0
    return 4.0 / 3.0 * Estar * math.sqrt(Rstar) * delta ** 1.5


@block("contact", "hertz", "sphere", "stress", source="Roark 7e Table 14.1 case 1")
def hertz_sphere_stress(F, Estar, Rstar, nu=0.3):
    """Contact radius a, peak pressure p0 = 3F/(2 pi a^2), max shear ~0.31 p0 at depth 0.48 a (nu=0.3).
    Returns dict(a, p0, tau_max, z_tau, delta)."""
    a = (3 * F * Rstar / (4 * Estar)) ** (1 / 3)
    p0 = 3 * F / (2 * math.pi * a * a)
    return dict(a=a, p0=p0, tau_max=0.31 * p0, z_tau=0.48 * a, delta=a * a / Rstar)


@block("contact", "hertz", "cylinder", "line_stress", source="Roark 7e Table 14.1 case 2",
       tags=("roller", "pin", "edge contact"))
def hertz_cylinder_stress(F_per_L, Estar, Rstar):
    """Parallel cylinders / cylinder on flat: half width b = sqrt(4 F' R*/(pi E*)), p0 = 2F'/(pi b)."""
    b = math.sqrt(4 * F_per_L * Rstar / (math.pi * Estar))
    return dict(b=b, p0=2 * F_per_L / (math.pi * b), tau_max=0.30 * 2 * F_per_L / (math.pi * b), z_tau=0.786 * b)


@block("contact", "hertz", "sphere", "impact", source="Johnson, Contact Mechanics sec. 11.4",
       tags=("contact time", "dt check", "peak force"))
def hertz_impact(m_star, v, Estar, Rstar):
    """Elastic sphere impact: peak indentation, peak force, contact duration. m* = m1 m2/(m1+m2)."""
    d_max = (15 * m_star * v * v / (16 * Estar * math.sqrt(Rstar))) ** 0.4
    F_max = 4 / 3 * Estar * math.sqrt(Rstar) * d_max ** 1.5
    t_c = 2.9432 * d_max / v       # = 2.868 (m*^2/(R* E*^2 v))^(1/5)
    return dict(delta_max=d_max, F_max=F_max, t_contact=t_c)


@block("contact", "hertz", "flat_punch", "normal_stiffness", source="Boussinesq (Johnson eq. 3.38)",
       tags=("rigid punch", "pad"))
def flat_punch_stiffness(a, Estar):
    """Rigid circular flat punch of radius a on a half-space: k = 2 a E*."""
    return 2 * a * Estar


@block("contact", "indentation", "sphere_tip", "plastic_table", kind="builder",
       source=("flat-punch stiffness 2 a E* (Johnson eq. 3.38) then fully plastic mean pressure 3Y over "
               "a^2 = 2 R delta_p (Johnson, Contact Mechanics ch. 6), contact radius capped at R"),
       tags=("tooth", "dent", "gouge", "plastic contact"))
def plastic_indentation_table(R, Y, Estar, depth_max=0.03, n=301, A_max=None):
    """Elastoplastic contact spring for a rounded tooth (corner radius R) pressing into a surface of yield Y:
    k = 2 R E*, yield force F(dp) = 3 Y A(dp), contact area A = pi min(2 R dp, R^2). A_max (optional): the tooth's
    whole bearing face (say tip length x blade thickness); the contact keeps growing past the corner, at the same
    rate, until that face is fully in. Without it the force stops rising once the corner is buried.
    Returns dict(k, acc_tab, F_tab, F_flat)."""
    import numpy as np
    acc = np.linspace(0.0, depth_max, n)
    cap = R * R if A_max is None else max(float(A_max) / math.pi, R * R)
    a2 = np.minimum(2.0 * R * acc, cap)
    F = 3.0 * Y * math.pi * np.maximum(a2, (0.1 * R) ** 2)
    return dict(k=2.0 * R * Estar, acc_tab=acc, F_tab=np.ascontiguousarray(F), F_flat=3.0 * Y * math.pi * R * R)

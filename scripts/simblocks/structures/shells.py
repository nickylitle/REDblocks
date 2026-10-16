"""Pressure vessels, thick cylinders (Lame), and interference fits.
Roark 7th ed. Table 13.1 (thin shells) and Table 13.5 (thick-walled cylinders); Shigley ch. 3."""
import math

from ..registry import block


@block("structures", "shells", "thin_cylinder", "internal_pressure", source="Roark 7e Table 13.1 case 1c",
       tags=("hoop", "pneumatic tank"))
def thin_cylinder_stress(p, r, t):
    """Thin cylinder hoop and axial stress (closed ends): (p r/t, p r/2t)."""
    return p * r / t, p * r / (2 * t)


@block("structures", "shells", "thin_sphere", "internal_pressure", source="Roark 7e Table 13.1 case 3a")
def thin_sphere_stress(p, r, t):
    """Thin sphere membrane stress p r / 2t."""
    return p * r / (2 * t)


@block("structures", "shells", "thick_cylinder", "pressure", source="Lame; Roark 7e Table 13.5 case 1",
       tags=("bore", "hub"))
def lame_stresses(pi, po, ri, ro, r):
    """Thick cylinder radial and hoop stress at r for internal pi and external po pressure. Returns (sr, st)."""
    A = (pi * ri * ri - po * ro * ro) / (ro * ro - ri * ri)
    B = (pi - po) * ri * ri * ro * ro / (ro * ro - ri * ri)
    return A - B / (r * r), A + B / (r * r)


@block("structures", "shells", "interference_fit", "press_fit", source="Shigley 3-56 (general materials)",
       tags=("press fit", "hub", "bearing seat", "shaft"))
def press_fit_pressure(delta_r, R, ri, ro, E_in, nu_in, E_out, nu_out):
    """Interface pressure from RADIAL interference delta_r at nominal radius R.
    Inner member: bore ri (0 for solid shaft) to R; outer member R to ro."""
    outer = (ro * ro + R * R) / (ro * ro - R * R) + nu_out
    inner = (R * R + ri * ri) / (R * R - ri * ri) - nu_in
    return delta_r / (R * (outer / E_out + inner / E_in))


@block("structures", "shells", "interference_fit", "torque_capacity", source="Coulomb friction on fit area")
def press_fit_torque(p, R, L, mu):
    """Slip torque of an interference fit: T = 2 pi mu p R^2 L."""
    return 2 * math.pi * mu * p * R * R * L


@block("structures", "shells", "interference_fit", "spin_loss", source="Lame + rotating disk, first order",
       tags=("centrifugal loosening",))
def press_fit_spin_growth(rho, nu, E, omega, R, ro):
    """Free radial growth of a spinning hub bore at R (annular disk), to subtract from interference:
    u = R/E * sigma_t(R) with sigma_t from the annular-disk spin solution (sigma_r(R) = 0)."""
    st = (3 + nu) / 4 * rho * omega ** 2 * (ro * ro + (1 - nu) / (3 + nu) * R * R)
    return R * st / E

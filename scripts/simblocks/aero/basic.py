"""Aerodynamics seed set: atmosphere, nondimensional groups, drag, skin friction, rotating-disk windage,
lift, and momentum-theory propulsion. Sources: Anderson 'Fundamentals of Aerodynamics'; Hoerner
'Fluid-Dynamic Drag'; Clift, Grace & Weber 1978; Daily & Nece 1960; ISA (ICAO Doc 7488)."""
import math

from ..registry import block
from .._jit import kernel


@block("aero", "atmosphere", "isa", "troposphere", source="ICAO Standard Atmosphere (Doc 7488)",
       tags=("altitude", "density"))
def isa_troposphere(h):
    """ISA temperature (K), pressure (Pa), density (kg/m^3) for 0 <= h <= 11 km."""
    T = 288.15 - 0.0065 * h
    p = 101325.0 * (T / 288.15) ** 5.25588
    return T, p, p / (287.058 * T)


@block("aero", "groups", "general", "reynolds", source="definition")
def reynolds(rho, v, L, mu):
    """Reynolds number rho v L / mu."""
    return rho * v * L / mu


@block("aero", "groups", "general", "dynamic_pressure", source="definition")
def dynamic_pressure(rho, v):
    """q = rho v^2 / 2."""
    return 0.5 * rho * v * v


@block("aero", "groups", "general", "mach", source="a = sqrt(gamma R T)")
def mach(v, T, gamma=1.4, R=287.058):
    """Mach number v / sqrt(gamma R T)."""
    return v / math.sqrt(gamma * R * T)


@block("aero", "drag", "bluff_body", "quadratic", kind="kernel", source="F = q Cd A, signed",
       tags=("air drag", "projectile", "debris"))
@kernel
def drag_force(v, rho, Cd, A):
    """Quadratic drag opposing velocity component v: -0.5 rho Cd A v |v|."""
    return -0.5 * rho * Cd * A * v * abs(v)


CD_TABLE = {   # typical subcritical values, Hoerner 'Fluid-Dynamic Drag' ch. 3 / Anderson Fig. 1.54
    "sphere": 0.47, "hemisphere_convex": 0.42, "hemisphere_concave": 1.42, "cube_face": 1.05,
    "cube_edge": 0.80, "flat_plate_normal": 1.28, "disk_normal": 1.17, "cylinder_crossflow": 1.17,
    "cylinder_axial_L/D=1": 0.90, "streamlined_body": 0.04, "wedge_30deg": 0.50, "person_standing": 1.2,
}


@block("aero", "drag", "sphere", "cd_vs_re", source="Clift, Grace & Weber 1978 (Re < 3e5)",
       tags=("particle", "ball"))
def cd_sphere(Re):
    """Sphere drag coefficient vs Reynolds number (Schiller-Naumann + Clift correction)."""
    if Re <= 0:
        return math.inf
    return 24 / Re * (1 + 0.15 * Re ** 0.687) + 0.42 / (1 + 42500 * Re ** -1.16)


@block("aero", "skin_friction", "flat_plate", "average", source="Blasius (laminar), Prandtl 1/5-power (turbulent)",
       tags=("boundary layer",))
def flat_plate_cf(Re_L, turbulent=None):
    """Average skin-friction coefficient over a plate of length L. Transition at Re 5e5 unless forced."""
    turb = Re_L > 5e5 if turbulent is None else turbulent
    return 0.074 * Re_L ** -0.2 if turb else 1.328 / math.sqrt(Re_L)


@block("aero", "windage", "rotating_disk", "torque", source="Daily & Nece 1960 / von Karman (free disk, both faces)",
       tags=("rotor", "spin-down", "aero loss"))
def disk_windage_torque(omega, R, rho, mu):
    """Aerodynamic drag torque on a free disk spinning in still air (both faces), N.m:
    M = 0.5 Cm rho w^2 R^5, Cm = 3.87 Re^-0.5 (laminar), 0.146 Re^-0.2 (turbulent, Re > 3e5), Re = w R^2 / nu."""
    Re = rho * omega * R * R / mu
    Cm = 3.87 / math.sqrt(Re) if Re < 3e5 else 0.146 * Re ** -0.2
    return 0.5 * Cm * rho * omega * omega * R ** 5


@block("aero", "windage", "bar_spinner", "torque", kind="kernel", source="strip theory with flat_plate_normal Cd",
       tags=("rotating bar", "aero loss"))
@kernel
def bar_windage_torque(omega, R, h, rho, Cd=1.28):
    """Strip-theory drag torque of a bar of half-length R and height h spinning about its center
    (two arms): M = 2 * integral 0.5 rho Cd h (w r)^2 r dr = rho Cd h w^2 R^4 / 4."""
    return rho * Cd * h * omega * omega * R ** 4 / 4.0


@block("aero", "lift", "airfoil", "thin_airfoil", source="Anderson 4.8 (thin airfoil theory)")
def thin_airfoil_cl(alpha, alpha_L0=0.0):
    """Section lift coefficient 2 pi (alpha - alpha_L0), radians."""
    return 2 * math.pi * (alpha - alpha_L0)


@block("aero", "lift", "finite_wing", "lift_slope", source="Helmbold 1942 (low aspect ratio valid)",
       tags=("wing", "aspect ratio"))
def finite_wing_lift_slope(AR, a0=2 * math.pi, sweep=0.0):
    """3-D lift-curve slope (per rad), Helmbold/DATCOM form."""
    k = a0 * math.cos(sweep) / (math.pi * AR)
    return a0 * math.cos(sweep) / (math.sqrt(1 + k * k) + k)


@block("aero", "drag", "finite_wing", "induced", source="Prandtl lifting line (Anderson 5.3)")
def induced_drag_cd(CL, AR, e=0.85):
    """Induced drag coefficient CL^2 / (pi e AR)."""
    return CL * CL / (math.pi * e * AR)


@block("aero", "propulsion", "actuator_disk", "hover_power", source="Rankine-Froude momentum theory",
       tags=("propeller", "rotor", "thrust"))
def ideal_hover_power(T, rho, A_disk):
    """Ideal induced power to produce thrust T statically: P = T^1.5 / sqrt(2 rho A)."""
    return T ** 1.5 / math.sqrt(2 * rho * A_disk)


@block("aero", "propulsion", "propeller", "thrust_coefficient", source="CT, CP definitions (n in rev/s)")
def propeller_thrust(CT, rho, n_rps, D):
    """Thrust from thrust coefficient: T = CT rho n^2 D^4."""
    return CT * rho * n_rps ** 2 * D ** 4


@block("aero", "drag", "bluff_body", "terminal_velocity", source="mg = q Cd A")
def terminal_velocity(m, g, rho, Cd, A):
    """Terminal velocity sqrt(2 m g / (rho Cd A))."""
    return math.sqrt(2 * m * g / (rho * Cd * A))

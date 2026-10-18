"""Aerodynamics seed set: atmosphere, nondimensional groups, drag, skin friction, rotating-disk windage,
lift, and momentum-theory propulsion. Sources: Anderson 'Fundamentals of Aerodynamics'; Hoerner
'Fluid-Dynamic Drag'; Clift, Grace & Weber 1978; Daily & Nece 1960; ISA (ICAO Doc 7488)."""
import math

from ..registry import block
from .._jit import kernel


@block("aero", "atmosphere", "isa", "troposphere", source="ICAO Standard Atmosphere (Doc 7488)",
       tags=("altitude", "density"))
def isa_troposphere(h):
    """ISA temperature (K), pressure (Pa), density (kg/m^3) at geopotential altitude h, -0.6 to 32 km: the troposphere (lapse 6.5 K/km to
    11 km), the isothermal tropopause layer (216.65 K, 11 to 20 km) and the lower stratosphere (+1 K/km, 20 to 32 km)."""
    if not -610.0 <= h <= 32000.0:
        raise ValueError("isa_troposphere covers -0.6 to 32 km; above that use the US76 atmosphere (gravity: "
                         "{planet: earth}) or another model")
    R, g = 287.058, 9.80665
    if h <= 11000.0:
        T = 288.15 - 0.0065 * h
        return T, 101325.0 * (T / 288.15) ** 5.25588, 101325.0 * (T / 288.15) ** 5.25588 / (R * T)
    p11 = 22632.06
    if h <= 20000.0:
        T = 216.65; p = p11 * math.exp(-g * (h - 11000.0) / (R * T))
        return T, p, p / (R * T)
    p20 = p11 * math.exp(-g * 9000.0 / (R * 216.65)); T = 216.65 + 0.001 * (h - 20000.0)
    p = p20 * (T / 216.65) ** (-g / (R * 0.001))
    return T, p, p / (R * T)


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
    if Re > 3e5:
        raise ValueError(f"cd_sphere is valid to Re 3e5 (got {Re:.3g}): past that the drag crisis drops Cd to about "
                         "0.1-0.2 depending on surface roughness; give Cd directly")
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


@block("aero", "propulsion", "propeller", "thrust_coefficient",
       source="CT, CP definitions (n in rev/s); advance ratio J = V/(n D) with the linear CT(J) fit "
              "(McCormick, Aerodynamics, Aeronautics and Flight Mechanics, ch. 6)",
       tags=("propeller", "thrust", "advance ratio", "airspeed", "forward flight"))
def propeller_thrust(CT, rho, n_rps, D, V=0.0, pitch=None, J0=None):
    """Propeller thrust T = CT(J) rho n^2 D^4, falling off with the airspeed through the disk.

    CT     static thrust coefficient (CT at J = 0, from the maker's data or a test stand)
    V      axial airspeed into the propeller (m/s): flight speed plus any headwind, 0 on a test stand
    pitch  geometric pitch (m), the distance one turn would screw forward; sets J0 = pitch/D if J0 is not given
    J0     advance ratio at which thrust reaches zero (no unit), about pitch/D for most fixed-pitch props
    CT(J) = CT (1 - J/J0) with J = V/(n D). Past J0 the propeller windmills and thrust turns negative (drag).
    With V = 0 this is the static thrust and pitch/J0 are not needed."""
    T0 = CT * rho * n_rps ** 2 * D ** 4
    if not V:
        return T0
    if J0 is None:
        if pitch is None:
            raise ValueError("propeller_thrust with an airspeed V needs pitch (m) or J0 (thrust-zero advance ratio)")
        J0 = pitch / D
    if J0 <= 0 or n_rps <= 0:
        raise ValueError("propeller_thrust needs J0 > 0 and n_rps > 0 when V is given")
    J = V / (n_rps * D)
    return T0 * (1.0 - J / J0)


@block("aero", "drag", "bluff_body", "terminal_velocity", source="mg = q Cd A")
def terminal_velocity(m, g, rho, Cd, A):
    """Terminal velocity sqrt(2 m g / (rho Cd A))."""
    return math.sqrt(2 * m * g / (rho * Cd * A))

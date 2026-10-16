"""Heat transfer seed set: resistances, convection correlations, radiation, lumped transient kernels,
motor winding heating, and plastic-work heating. Sources: Incropera & DeWitt 'Fundamentals of Heat and
Mass Transfer' (eq. numbers 7th ed.); Taylor & Quinney 1934."""
import math

from ..registry import block
from .._jit import kernel

SIGMA_SB = 5.670374419e-8


# ---------------------------------------------------------------- resistances
@block("thermal", "conduction", "plane_wall", "resistance", source="Incropera 3.6")
def R_plane_wall(L, k, A):
    """Conduction resistance L / (k A), K/W."""
    return L / (k * A)


@block("thermal", "conduction", "cylinder_shell", "resistance", source="Incropera 3.33", tags=("motor can", "pipe"))
def R_cylinder(ri, ro, k, L):
    """Radial conduction resistance ln(ro/ri) / (2 pi k L)."""
    return math.log(ro / ri) / (2 * math.pi * k * L)


@block("thermal", "conduction", "sphere_shell", "resistance", source="Incropera 3.41")
def R_sphere(ri, ro, k):
    """Spherical shell resistance (1/ri - 1/ro) / (4 pi k)."""
    return (1 / ri - 1 / ro) / (4 * math.pi * k)


@block("thermal", "convection", "surface", "resistance", source="Newton cooling")
def R_convection(h, A):
    """Convection resistance 1 / (h A)."""
    return 1.0 / (h * A)


@block("thermal", "radiation", "surface", "linearized_h", source="Incropera 1.9")
def h_radiation(eps, T, T_sur):
    """Linearized radiation coefficient eps sigma (T + Tsur)(T^2 + Tsur^2), W/m^2K."""
    return eps * SIGMA_SB * (T + T_sur) * (T * T + T_sur * T_sur)


@block("thermal", "fins", "straight_fin", "efficiency", source="Incropera 3.89 (adiabatic tip, corrected length)",
       tags=("heatsink", "ESC"))
def fin_efficiency(h, k, t, L):
    """Straight rectangular fin efficiency tanh(m Lc)/(m Lc), m = sqrt(2h/(k t)), Lc = L + t/2."""
    m = math.sqrt(2 * h / (k * t))
    Lc = L + t / 2
    return math.tanh(m * Lc) / (m * Lc)


# ---------------------------------------------------------------- convection correlations
@block("thermal", "convection", "flat_plate", "forced_average", source="Incropera 7.30 / 7.38 (mixed)",
       tags=("external flow",))
def nu_flat_plate(Re_L, Pr, Re_crit=5e5):
    """Average Nusselt number over a flat plate, laminar or mixed laminar-turbulent."""
    if Re_L <= Re_crit:
        return 0.664 * math.sqrt(Re_L) * Pr ** (1 / 3)
    A = 0.037 * Re_crit ** 0.8 - 0.664 * math.sqrt(Re_crit)
    return (0.037 * Re_L ** 0.8 - A) * Pr ** (1 / 3)


@block("thermal", "convection", "cylinder", "crossflow", source="Churchill & Bernstein 1977 (Incropera 7.54)",
       tags=("motor in airflow", "wire"))
def nu_cylinder_crossflow(Re, Pr):
    """Average Nu for a cylinder in crossflow, all Re*Pr > 0.2."""
    return 0.3 + 0.62 * Re ** 0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25 * \
        (1 + (Re / 282000) ** 0.625) ** 0.8


@block("thermal", "convection", "vertical_plate", "natural", source="Churchill & Chu 1975 (Incropera 9.26)")
def nu_vertical_plate_natural(Ra, Pr):
    """Average Nu, natural convection on a vertical plate, full Ra range."""
    return (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2


@block("thermal", "convection", "horizontal_cylinder", "natural", source="Churchill & Chu 1975 (Incropera 9.34)",
       tags=("motor at rest",))
def nu_horizontal_cylinder_natural(Ra, Pr):
    """Average Nu, natural convection on a horizontal cylinder, Ra < 1e12."""
    return (0.60 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2


@block("thermal", "convection", "pipe", "turbulent", source="Gnielinski 1976 (Incropera 8.62)",
       tags=("internal flow", "coolant"))
def nu_pipe_gnielinski(Re, Pr):
    """Fully developed turbulent pipe Nu (3000 < Re < 5e6), smooth-pipe Petukhov friction factor."""
    f = (0.790 * math.log(Re) - 1.64) ** -2
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


@block("thermal", "convection", "general", "rayleigh", source="definition")
def rayleigh(g, beta, dT, L, nu_kin, alpha_th):
    """Rayleigh number g beta dT L^3 / (nu alpha)."""
    return g * beta * dT * L ** 3 / (nu_kin * alpha_th)


# ---------------------------------------------------------------- transient
@block("thermal", "transient", "lumped", "biot", source="Incropera 5.10")
def biot_number(h, Lc, k):
    """Biot number h Lc / k (lumped valid for Bi < 0.1, Lc = V/A)."""
    return h * Lc / k


@block("thermal", "transient", "lumped", "analytic", source="Incropera 5.6")
def lumped_temperature(t, T0, T_inf, h, A, m, cp, q_gen=0.0):
    """Lumped body temperature with convection and constant internal heating q_gen (W)."""
    tau = m * cp / (h * A)
    T_ss = T_inf + q_gen / (h * A)
    return T_ss + (T0 - T_ss) * math.exp(-t / tau)


@block("thermal", "transient", "node", "rate", kind="kernel", source="energy balance m cp dT/dt = sum Q",
       tags=("thermal network", "explicit"))
@kernel
def node_dTdt(T, Q_in, T_amb, R_to_amb, m, cp):
    """dT/dt for one thermal node with heat input Q_in and a resistance R to ambient."""
    return (Q_in - (T - T_amb) / R_to_amb) / (m * cp)


@block("thermal", "transient", "node", "link_flow", kind="kernel", source="Fourier / Newton")
@kernel
def link_heat_flow(T_a, T_b, R):
    """Heat flow a -> b through resistance R (W)."""
    return (T_a - T_b) / R


@block("thermal", "electrical", "winding", "resistance_vs_T", kind="kernel", source="R = R0 (1 + a (T - T0))",
       tags=("copper", "motor", "derating"))
@kernel
def copper_resistance(R0, T, T0, alpha_R):
    """Temperature-dependent winding resistance (alpha_R = 0.00393 /K for copper)."""
    return R0 * (1.0 + alpha_R * (T - T0))


@block("thermal", "deformation", "plastic_work", "adiabatic_heating", source="Taylor & Quinney 1934 (beta ~ 0.9)",
       tags=("impact", "shear band", "temperature rise"))
def plastic_heating(W_plastic, m, cp, beta=0.9):
    """Adiabatic temperature rise from plastic work: dT = beta W / (m cp)."""
    return beta * W_plastic / (m * cp)


@block("thermal", "friction", "sliding", "heat_partition", source="Blok partition for semi-infinite bodies",
       tags=("skid", "brake"))
def friction_heat_partition(k1, rho1, c1, k2, rho2, c2):
    """Fraction of frictional heat entering body 1: e1/(e1+e2), effusivity e = sqrt(k rho c)."""
    e1, e2 = math.sqrt(k1 * rho1 * c1), math.sqrt(k2 * rho2 * c2)
    return e1 / (e1 + e2)


# ---------------------------------------------------------------- generic entry points
@block("thermal", "conduction", "general", "resistance",
       source="Incropera 3.6 (plane wall), 3.33 (cylinder), 3.41 (sphere); Newton cooling",
       tags=("thermal network", "motor can", "pipe", "wall", "convection"))
def thermal_resistance(kind, k=None, A=None, L=None, ri=None, ro=None, h=None):
    """Thermal resistance (K/W) of one path in a thermal network.

    kind
      plane_wall  L / (k A):                    thickness L, conductivity k, area A
      cylinder    ln(ro/ri) / (2 pi k L):       radial conduction through a shell of length L
      sphere      (1/ri - 1/ro) / (4 pi k):     spherical shell
      convection  1 / (h A):                    surface to fluid, film coefficient h
    Returns R (K/W). Resistances in series add; use it with node_dTdt style networks or lumped_temperature."""
    key = str(kind).strip().lower()
    req = {"plane_wall": ("L", "k", "A"), "cylinder": ("ri", "ro", "k", "L"), "sphere": ("ri", "ro", "k"),
           "convection": ("h", "A")}
    if key not in req:
        raise ValueError(f"thermal_resistance: unknown kind {kind!r}; use one of {', '.join(req)}")
    v = dict(k=k, A=A, L=L, ri=ri, ro=ro, h=h)
    miss = [n for n in req[key] if v[n] is None]
    if miss:
        raise ValueError(f"thermal_resistance '{key}' needs {', '.join(miss)}")
    if key == "plane_wall":
        return R_plane_wall(L, k, A)
    if key == "cylinder":
        return R_cylinder(ri, ro, k, L)
    if key == "sphere":
        return R_sphere(ri, ro, k)
    return R_convection(h, A)


@block("thermal", "convection", "general", "nusselt",
       source="Incropera 7.30/7.38 (flat plate), Churchill & Bernstein 1977, Churchill & Chu 1975 (vertical plate, "
              "horizontal cylinder), Gnielinski 1976",
       tags=("external flow", "internal flow", "natural convection", "film coefficient", "motor", "coolant"))
def nusselt(case, Pr, Re=None, Ra=None, Re_crit=5e5):
    """Average Nusselt number for a convection case; h = Nu k_fluid / L.

    case
      flat_plate                   Re (on plate length L), Pr; laminar or mixed laminar-turbulent past Re_crit
      cylinder_crossflow           Re (on diameter), Pr; Churchill-Bernstein, Re Pr > 0.2
      vertical_plate_natural       Ra (on height), Pr; Churchill-Chu, full Ra range
      horizontal_cylinder_natural  Ra (on diameter), Pr; Churchill-Chu, Ra < 1e12
      pipe_gnielinski              Re (on diameter), Pr; fully developed turbulent, 3000 < Re < 5e6
    Returns Nu."""
    key = str(case).strip().lower()
    forced = {"flat_plate", "cylinder_crossflow", "pipe_gnielinski"}
    natural = {"vertical_plate_natural", "horizontal_cylinder_natural"}
    if key in forced:
        if Re is None:
            raise ValueError(f"nusselt '{key}' needs Re")
        if key == "flat_plate":
            return nu_flat_plate(Re, Pr, Re_crit)
        return nu_cylinder_crossflow(Re, Pr) if key == "cylinder_crossflow" else nu_pipe_gnielinski(Re, Pr)
    if key in natural:
        if Ra is None:
            raise ValueError(f"nusselt '{key}' needs Ra (see rayleigh)")
        return nu_vertical_plate_natural(Ra, Pr) if key == "vertical_plate_natural" \
            else nu_horizontal_cylinder_natural(Ra, Pr)
    raise ValueError(f"nusselt: unknown case {case!r}; use one of {', '.join(sorted(forced | natural))}")

"""Machine elements, closed form: shaft whirl, rolling bearings, press fits, keys, set screws and clamp collars,
helical springs, belt and chain drives, cracks and notches, bonded and welded joints. SI in and out (rpm where
named). Shigley's Mechanical Engineering Design 10e unless a block says otherwise."""
import math

import numpy as np

from ..registry import block

SH = "Shigley 10e"


# ---------------------------------------------------------------------------------------------- shafts
def _ss_influence(L, x, a, EI):
    """Deflection at x of a simply supported span from a unit load at a (beam table)."""
    b = L - a
    if x <= a:
        return b * x * (L * L - b * b - x * x) / (6 * EI * L)
    return a * (L - x) * (2 * L * x - x * x - a * a) / (6 * EI * L)


@block("structures", "machine", "shaft", "critical_speed",
       source=SH + " sec. 7-6: Rayleigh (eq. 7-23) upper bound and Dunkerley (eq. 7-24) lower bound",
       tags=("whirl", "rotor", "resonance", "rpm"))
def shaft_critical_speed(E, d, L, rho=7850.0, masses=(), positions=(), supports="simply_supported", d_inner=0.0):
    """First whirl (critical) speed of a uniform shaft on two bearings carrying point masses (discs, hubs).
    supports: simply_supported (masses between the bearings, positions from one bearing) or cantilever
    (positions from the fixed bearing). Returns dict(rpm, rpm_rayleigh, rpm_dunkerley, rpm_shaft_alone)."""
    I = math.pi * (d ** 4 - d_inner ** 4) / 64; A = math.pi * (d * d - d_inner * d_inner) / 4; EI = E * I
    mL = rho * A
    lam = math.pi if supports == "simply_supported" else 1.8751
    w_s = (lam / L) ** 2 * math.sqrt(EI / mL)
    ms = [float(m) for m in masses]; xs = [float(p) for p in positions]
    if len(ms) != len(xs):
        raise ValueError("give one position for each mass")
    if any(not 0.0 <= x <= L for x in xs):
        raise ValueError(f"every mass position is along the shaft, 0 to L = {L:g} m from "
                         f"{'the first bearing' if supports == 'simply_supported' else 'the fixed bearing'}")
    if not ms:
        rpm = w_s * 60 / (2 * math.pi)
        return dict(rpm=rpm, rpm_rayleigh=rpm, rpm_dunkerley=rpm, rpm_shaft_alone=rpm)
    n = len(ms)
    if supports == "simply_supported":
        a = np.array([[_ss_influence(L, xs[i], xs[j], EI) for j in range(n)] for i in range(n)])
    else:                                       # cantilever: y(x) from a unit load at p
        def cant(x, p):
            return x * x * (3 * p - x) / (6 * EI) if x <= p else p * p * (3 * x - p) / (6 * EI)
        a = np.array([[cant(xs[i], xs[j]) for j in range(n)] for i in range(n)])
    g = 9.81
    y = a @ (np.array(ms) * g)                  # static deflection under all the weights together
    w_r = math.sqrt(g * float(np.dot(ms, y)) / float(np.dot(ms, y * y)))
    w_r = 1.0 / math.sqrt(1.0 / w_r ** 2 + 1.0 / w_s ** 2)        # the shaft's own mass lowers it (Dunkerley sum)
    w_d = 1.0 / math.sqrt(1.0 / w_s ** 2 + sum(m * a[i, i] for i, m in enumerate(ms)))
    k = 60 / (2 * math.pi)
    return dict(rpm=0.5 * (w_r + w_d) * k, rpm_rayleigh=w_r * k, rpm_dunkerley=w_d * k, rpm_shaft_alone=w_s * k)


@block("structures", "machine", "rolling_bearing", "life", source="ISO 281: L10 = (C/P)^p 1e6 rev, a1 for reliability",
       tags=("ball", "roller", "L10", "bearing"))
def bearing_life(C, P, rpm, kind="ball", reliability=0.9):
    """Rated life of a rolling bearing: p = 3 ball, 10/3 roller. reliability 0.90 to 0.99 (a1 factor).
    Returns dict(L_rev, L_hours, a1)."""
    p = 3.0 if kind == "ball" else 10.0 / 3.0
    A1 = {0.90: 1.0, 0.95: 0.64, 0.96: 0.55, 0.97: 0.47, 0.98: 0.37, 0.99: 0.25}
    a1 = A1.get(round(reliability, 2))
    if a1 is None:
        raise ValueError("reliability is one of 0.90, 0.95, 0.96, 0.97, 0.98, 0.99")
    L = a1 * (C / P) ** p * 1e6
    return dict(L_rev=L, L_hours=L / (rpm * 60) if rpm > 0 else math.inf, a1=a1)


@block("structures", "machine", "rolling_bearing", "static", source="ISO 76: s0 = C0 / P0 (shock loads want s0 >= 2)",
       tags=("bearing", "brinell", "impact"))
def bearing_static_check(C0, Fr, Fa=0.0, X0=0.6, Y0=0.5):
    """Static safety of a rolling bearing under a peak load (a hit): P0 = max(X0 Fr + Y0 Fa, Fr). Defaults for a
    deep-groove ball bearing. s0 below 1 brinells the races. Returns dict(P0, s0)."""
    P0 = max(X0 * Fr + Y0 * Fa, Fr)
    return dict(P0=P0, s0=C0 / P0 if P0 > 0 else math.inf)


# ---------------------------------------------------------------------------------------------- shaft-hub joints
@block("structures", "machine", "press_fit", "interference", source=SH + " eqs. 3-56 to 3-58 (Lame thick cylinders)",
       tags=("hub", "shrink fit", "slip torque", "interference"))
def press_fit(delta, d, D_hub, L, E_hub, E_shaft, nu_hub=0.3, nu_shaft=0.3, mu=0.15, d_shaft_bore=0.0):
    """Interference fit of a shaft (diameter d, optional bore) in a hub (outer diameter D_hub, length L). delta is
    the diametral interference: shaft diameter minus hub bore, as fit tables give it.
    Returns dict(p, hub_hoop_bore, shaft_hoop, slip_torque, push_force)."""
    r = d / 2; ro = D_hub / 2; ri = d_shaft_bore / 2; dr = delta / 2
    hub = (ro * ro + r * r) / (ro * ro - r * r) + nu_hub
    shaft = (r * r + ri * ri) / (r * r - ri * ri) - nu_shaft
    p = dr / (r / E_hub * hub + r / E_shaft * shaft)
    return dict(p=p, hub_hoop_bore=p * (ro * ro + r * r) / (ro * ro - r * r),
                shaft_hoop=-p * (r * r + ri * ri) / (r * r - ri * ri),
                slip_torque=math.pi * mu * p * d * d * L / 2, push_force=math.pi * mu * p * d * L)


@block("structures", "machine", "key", "torque", source=SH + " sec. 7-7: shear Ssy = 0.577 Sy, crushing on h/2",
       tags=("keyway", "hub", "shaft", "square key"))
def key_capacity(d, w, h, L, Sy):
    """Torque a parallel key carries before it shears (across w L) or crushes (on h/2 L). Returns dict(T_shear,
    T_crush, T)."""
    Ts = 0.577 * Sy * w * L * d / 2
    Tc = Sy * (h / 2) * L * d / 2
    return dict(T_shear=Ts, T_crush=Tc, T=min(Ts, Tc))


SET_SCREW_LBF = [(0.060, 50), (0.073, 65), (0.086, 85), (0.099, 120), (0.112, 160), (0.125, 200), (0.138, 250),
                 (0.164, 385), (0.190, 540), (0.250, 1000), (0.3125, 1500), (0.375, 2000), (0.4375, 2500),
                 (0.5, 3000), (0.625, 4000), (0.75, 5000)]


@block("structures", "machine", "set_screw", "holding", source=SH + " Table 7-4 (cup point, seated to the table torque)",
       tags=("grub screw", "hub", "collar"))
def set_screw_holding(d_screw, d_shaft, n=1, factor=1.0):
    """Holding force and torque of cup-point set screws (diameter d_screw) on a shaft. A second screw at 90 deg or
    less adds about the same again. factor scales the table (cone point ~1.1, flat ~0.9). Returns dict(F, T)."""
    din = d_screw / 0.0254
    xs = [x for x, _ in SET_SCREW_LBF]; ys = [y for _, y in SET_SCREW_LBF]
    if not xs[0] * 0.999 <= din <= xs[-1] * 1.001:
        raise ValueError(f"Shigley's table covers set screws from {xs[0]:g} to {xs[-1]:g} in (got {din:.3g} in)")
    F = float(np.interp(din, xs, ys)) * 4.448222 * factor * n
    return dict(F=F, T=F * d_shaft / 2)


@block("structures", "machine", "clamp_collar", "holding",
       source="uniform contact pressure: n Fi = p d L per half, T = mu p pi d L d / 2", tags=("shaft collar", "split hub"))
def clamp_collar_holding(Fi, d_shaft, n_screws=1, mu=0.2, halves=1):
    """Holding torque of a split clamp collar or clamping hub: n_screws per clamp, each at preload Fi.
    halves = 2 for a two-piece collar (screws on both sides). Returns dict(T, F_axial, p_line)."""
    N = n_screws * Fi * (2 if halves == 2 else 1)
    T = mu * N * math.pi * d_shaft / 2
    return dict(T=T, F_axial=mu * N * math.pi, p_line=N / d_shaft)


# ---------------------------------------------------------------------------------------------- springs
@block("structures", "machine", "helical_spring", "compression", source=SH + " eqs. 10-1 to 10-9, Table 10-1, eq. 10-25",
       tags=("coil", "wahl", "surge", "solid height"))
def helical_spring(d, D, n_active, G=79.3e9, ends="squared_ground", rho=7850.0, free_length=None):
    """A round-wire helical compression spring: d wire, D mean coil diameter. Returns dict(k, C, K_wahl,
    tau_per_N, solid_length, n_total, surge_hz, travel_to_solid)."""
    C = D / d
    KW = (4 * C - 1) / (4 * C - 4) + 0.615 / C
    k = G * d ** 4 / (8 * D ** 3 * n_active)
    extra = {"plain": 0, "plain_ground": 1, "squared": 2, "squared_ground": 2}[ends]
    Nt = n_active + extra
    Ls = d * (Nt + 1) if ends in ("plain", "squared") else d * Nt
    m = rho * math.pi ** 2 * d * d * D * n_active / 4
    out = dict(k=k, C=C, K_wahl=KW, tau_per_N=KW * 8 * D / (math.pi * d ** 3), solid_length=Ls, n_total=Nt,
               surge_hz=0.5 * math.sqrt(k / m))
    out["travel_to_solid"] = (free_length - Ls) if free_length else None
    return out


# ---------------------------------------------------------------------------------------------- drives
@block("structures", "machine", "belt", "slip", source="Euler capstan F1/F2 = exp(mu theta / sin(beta/2)); "
       + SH + " sec. 17-2 (centrifugal tension)", tags=("pulley", "v-belt", "timing", "drive"))
def belt_drive(Fi, d, wrap, mu=0.3, groove=math.pi, m_per_L=0.0, rpm=0.0):
    """A friction belt on a pulley (pitch diameter d, wrap angle in rad, groove angle in rad: pi = flat; a V-belt is
    about 36-40 deg, 0.63-0.70 rad). Fi is the installed (static) tension per strand. Running, both strands also carry
    the centrifugal tension Fc = m' v^2 (Shigley eq. 17-9: Fi = (F1 + F2) / 2 - Fc) and slip starts at
    (F1 - Fc) / (F2 - Fc) = exp(mu wrap / sin(groove / 2)). Returns dict(T_slip, F1, F2, ratio, Fc)."""
    if not 0.0 < groove <= math.pi + 1e-9:
        raise ValueError("groove is the groove angle in radians: pi for a flat belt, about 0.63-0.70 rad (36-40 deg) "
                         "for a V-belt")
    v = rpm * 2 * math.pi / 60 * d / 2
    Fc = m_per_L * v * v
    e = math.exp(mu * wrap / math.sin(groove / 2))
    F1 = Fc + 2 * Fi * e / (e + 1); F2 = Fc + 2 * Fi / (e + 1)
    return dict(T_slip=(F1 - F2) * d / 2, F1=F1, F2=F2, ratio=e, Fc=Fc)


@block("structures", "machine", "chain", "tension", source=SH + " sec. 17-5: pitch diameter, chordal speed variation",
       tags=("sprocket", "roller chain", "drive"))
def chain_drive(T, pitch, teeth, m_per_L=0.0, rpm=0.0):
    """Roller chain on a sprocket: pull from the torque plus centrifugal tension. Returns dict(D, F, Fc,
    chordal_speed_variation)."""
    D = pitch / math.sin(math.pi / teeth)
    v = rpm * 2 * math.pi / 60 * D / 2
    Fc = m_per_L * v * v
    return dict(D=D, F=T / (D / 2) + Fc, Fc=Fc, chordal_speed_variation=1 - math.cos(math.pi / teeth))


# ---------------------------------------------------------------------------------------------- cracks and notches
@block("structures", "fracture", "crack", "toughness", source="K = Y sigma sqrt(pi a) (Anderson, Fracture Mechanics 4e)",
       tags=("K_IC", "brittle", "hardened", "crack length"))
def crack_check(K_IC, sigma, a, geometry="edge"):
    """Linear-elastic fracture check: a crack of size a (half length for a center crack, depth for an edge or
    surface crack) under stress sigma. geometry: center (Y 1.0), edge (1.12), surface (semicircular, deepest point,
    0.73). Returns dict(K, SF, a_critical)."""
    Y = {"center": 1.0, "edge": 1.12, "surface": 0.73}[geometry]
    K = Y * sigma * math.sqrt(math.pi * a)
    return dict(K=K, SF=K_IC / K if K > 0 else math.inf, a_critical=(K_IC / (Y * sigma)) ** 2 / math.pi if sigma > 0 else math.inf)


@block("structures", "fatigue", "notch", "sensitivity", source=SH + " eqs. 6-32 to 6-35 (Neuber constant from Sut)",
       tags=("Kf", "Kt", "notch", "fillet"))
def notch_sensitivity(Kt, r, Sut, loading="bending"):
    """Fatigue notch factor from the theoretical one: Kf = 1 + q (Kt - 1), q = 1 / (1 + sqrt(a) / sqrt(r)).
    Steels, Sut 340 to 1700 MPa. loading: bending (also axial) or torsion. Returns dict(Kf, q)."""
    S = Sut / 6.894757e6                        # ksi
    if not 340e6 * 0.999 <= Sut <= 1700e6 * 1.001:
        raise ValueError(f"the Neuber constant fit covers steels with Sut 340 to 1700 MPa (got {Sut / 1e6:.0f} MPa); "
                         "for other materials use q from a chart (Shigley Fig. 6-20) or q = 1 (Kf = Kt, conservative)")
    if loading == "torsion":
        sa = 0.190 - 2.51e-3 * S + 1.35e-5 * S * S - 2.67e-8 * S ** 3
    else:
        sa = 0.246 - 3.08e-3 * S + 1.51e-5 * S * S - 2.67e-8 * S ** 3
    sa = max(sa, 0.0) * math.sqrt(0.0254)       # sqrt(in) -> sqrt(m)
    q = 1.0 / (1.0 + sa / math.sqrt(r)) if r > 0 else 0.0
    return dict(Kf=1 + q * (Kt - 1), q=q)


# ---------------------------------------------------------------------------------------------- bonded and welded
@block("structures", "joints", "adhesive", "lap_shear", source="Volkersen 1938 shear-lag (single lap, no peel)",
       tags=("glue", "epoxy", "bond", "bonded"))
def lap_joint_shear(P, L, b, t1, E1, t2, E2, G_a, t_a, tau_allow=None):
    """Bonded lap joint: peak adhesive shear at the overlap ends (the average understates it for a long overlap).
    Returns dict(tau_avg, tau_peak, peak_factor, P_allow)."""
    lam = math.sqrt(G_a / t_a * (1 / (E1 * t1) + 1 / (E2 * t2)))
    psi = min(E1 * t1, E2 * t2) / max(E1 * t1, E2 * t2)
    x = lam * L
    fac = x * (math.cosh(x) + psi) / ((1 + psi) * math.sinh(x))
    tavg = P / (b * L)
    return dict(tau_avg=tavg, tau_peak=fac * tavg, peak_factor=fac,
                P_allow=(tau_allow * b * L / fac) if tau_allow else None)


@block("structures", "joints", "weld", "fillet", source="throat 0.707 h; AISC allowable 0.3 x electrode strength (E70: 70 ksi)",
       tags=("weld", "fillet", "throat"))
def fillet_weld(h, L, F=None, S_electrode=482.6e6, n_welds=1):
    """Fillet weld in shear on its throat. Returns dict(throat_area, tau, F_allow, SF)."""
    A = 0.707 * h * L * n_welds
    allow = 0.3 * S_electrode
    tau = F / A if F else None
    return dict(throat_area=A, tau=tau, F_allow=allow * A, SF=(allow / tau) if tau else None)

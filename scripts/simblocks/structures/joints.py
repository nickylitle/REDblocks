"""Bolted and pinned joints (Shigley ch. 8; Roark 7e ch. 14 for bearing; Bickford for torque-preload)."""
import math

from ..registry import block


@block("structures", "joints", "bolt", "tensile_area", source="ISO 898 / ASME B1.1: As = pi/4 (d - 0.9382 p)^2",
       tags=("thread", "M3", "M4"))
def bolt_tensile_area(d, pitch):
    """Tensile stress area of a metric/UN thread."""
    return math.pi / 4 * (d - 0.9382 * pitch) ** 2


METRIC_COARSE_PITCH = {0.002: 0.0004, 0.0025: 0.00045, 0.003: 0.0005, 0.004: 0.0007, 0.005: 0.0008,
                       0.006: 0.001, 0.008: 0.00125, 0.010: 0.0015, 0.012: 0.00175}
PROPERTY_CLASS = {"8.8": (640e6, 800e6), "10.9": (940e6, 1040e6), "12.9": (1100e6, 1220e6),
                  "A2-70": (450e6, 700e6)}   # (proof-ish yield, ultimate) ISO 898-1 / ISO 3506


@block("structures", "joints", "bolt", "preload", source="T = K F d (Shigley 8-27), K ~ 0.2 dry, 0.15 lubed")
def bolt_preload_from_torque(T, d, K=0.2):
    """Bolt preload from tightening torque."""
    return T / (K * d)


@block("structures", "joints", "bolt", "shear", source="direct shear on shank or thread root")
def bolt_shear_capacity(d, tau_u, n_planes=1, threads_in_plane=False, pitch=None):
    """Ultimate shear capacity: tau_u * area * planes. Uses root area if threads are in the shear plane."""
    if threads_in_plane:
        A = bolt_tensile_area(d, pitch or METRIC_COARSE_PITCH.get(round(d, 4), 0.15 * d))
    else:
        A = math.pi * d * d / 4
    return tau_u * A * n_planes


@block("structures", "joints", "plate_hole", "bearing", source="Shigley 8-11", tags=("bearing", "elongation"))
def bearing_capacity(d, t, s_bearing):
    """Hole bearing capacity F = s_bearing d t (s_bearing ~ 1.5-2.0 su for metals edge distance >= 2d)."""
    return s_bearing * d * t


@block("structures", "joints", "plate_hole", "tear_out",
       source="2 tau t (e - d/2): MechaniCalc Lug Analysis (Melcon & Hoblit 1953; Bruhn 1973); AISC 360-16 J3-6c uses 1.2 l_c t F_u",
       tags=("shear out", "edge distance"))
def tear_out_capacity(e, d, t, tau_u):
    """Shear tear-out with edge distance e (hole center to edge): F = 2 tau_u t (e - d/2)."""
    return 2 * tau_u * t * (e - d / 2)


@block("structures", "joints", "plate_hole", "net_tension", source="net section")
def net_tension_capacity(w, d, t, su, n_holes=1):
    """Net-section tension capacity su (w - n d) t."""
    return su * (w - n_holes * d) * t


@block("structures", "joints", "bolt", "stiffness", source="AE/L")
def bolt_stiffness(d, L_grip, E, d_minor=None, L_thread=0.0):
    """Bolt axial stiffness, shank and threaded portions in series."""
    As = math.pi * d * d / 4
    At = math.pi * (d_minor or 0.85 * d) ** 2 / 4
    Ls = L_grip - L_thread
    return 1.0 / (Ls / (As * E) + L_thread / (At * E))


@block("structures", "joints", "member", "stiffness", source="Wileman et al. 1991 (Shigley 8-23)",
       tags=("clamped member", "joint constant"))
def member_stiffness_wileman(E, d, L_grip, A=0.78715, B=0.62873):
    """Clamped-member stiffness k_m = E d A exp(B d / l). Defaults are for steel."""
    return E * d * A * math.exp(B * d / L_grip)


@block("structures", "joints", "thread", "stripping", source="FED-STD-H28 approximate shear area",
       tags=("tapped hole", "aluminum threads"))
def thread_strip_capacity(d, L_engaged, tau_u_weaker, pitch):
    """Approximate stripping capacity of internal threads: tau_u * (pi d L_e * 0.75 approx engagement)."""
    return tau_u_weaker * math.pi * d * L_engaged * 0.75 * (1 - 0.5 * pitch / d)


@block("structures", "joints", "tab", "shear_out_depth", source="LEGACY: user-supplied impact model notes (no published basis found for the pi*M_p term; use blunt_punch_line_force + tear_out_capacity)",
       tags=("gouge", "depth hardening"))
def tab_shear_out_force(mp, tau_y, t, pen):
    """Gouge/tab shear-out flow force hardening with penetration depth: F = pi m_p + 2 tau_y t pen."""
    return math.pi * mp + 2.0 * tau_y * t * pen

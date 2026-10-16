"""Natural frequencies and dynamic load factors (Roark 7th ed. ch. 16, Table 16.1; Blevins)."""
import math

from ..registry import block

SRC = "Roark 7e Table 16.1 / Blevins"
BEAM_LAMBDA1 = {"cantilever": 1.8751, "simply_supported": math.pi, "fixed-fixed": 4.7300,
                "free-free": 4.7300, "fixed-pinned": 3.9266, "fixed-guided": 2.3650}


@block("structures", "vibration", "beam", "first_mode", source=SRC, tags=("ringing", "dt check"))
def beam_frequency(E, I, m_per_L, L, bc="cantilever", mode_lambda=None):
    """First bending natural frequency (Hz): f = lambda^2/(2 pi L^2) sqrt(EI/m)."""
    lam = mode_lambda or BEAM_LAMBDA1[bc]
    return lam ** 2 / (2 * math.pi * L * L) * math.sqrt(E * I / m_per_L)


@block("structures", "vibration", "cantilever", "tip_mass", source="Rayleigh, 0.2427 m L effective mass")
def cantilever_tip_mass_frequency(E, I, L, m_per_L, M_tip):
    """Cantilever with tip mass: f = (1/2pi) sqrt(3EI/L^3 / (M + 0.2427 m L))."""
    return math.sqrt(3 * E * I / L ** 3 / (M_tip + 0.2427 * m_per_L * L)) / (2 * math.pi)


PLATE_LAMBDA2 = {"circular_clamped": 10.216, "circular_ss": 4.977, "circular_free": 5.253,
                 "square_ss": 19.739, "square_clamped": 35.99}


@block("structures", "vibration", "plate", "first_mode", source=SRC + " (Leissa)")
def plate_frequency(D, rho, t, a, case="circular_clamped"):
    """Plate fundamental (Hz): f = lambda^2/(2 pi a^2) sqrt(D/(rho t)); a = radius (circular) or side (square)."""
    return PLATE_LAMBDA2[case] / (2 * math.pi * a * a) * math.sqrt(D / (rho * t))


@block("structures", "vibration", "spring_mass", "frequency", source="sqrt(k/m)")
def spring_mass_frequency(k, m):
    """Undamped natural frequency (rad/s)."""
    return math.sqrt(k / m)


@block("structures", "vibration", "spring_mass", "drop_impact", source="Roark 7e eq. 16.3-1 (energy method)",
       tags=("impact factor", "drop"))
def drop_impact_deflection(W, k, h):
    """Peak deflection from a weight W dropped height h onto a spring: d_st (1 + sqrt(1 + 2h/d_st))."""
    dst = W / k
    return dst * (1 + math.sqrt(1 + 2 * h / dst))


@block("structures", "vibration", "spring_mass", "moving_mass_impact", source="energy: m v^2/2 = k d^2/2",
       tags=("impact", "horizontal"))
def moving_mass_impact(m, v, k):
    """Peak deflection and force when mass m at speed v hits a linear spring k: (d, F)."""
    d = v * math.sqrt(m / k)
    return d, k * d


@block("structures", "vibration", "bar", "wave_speed", source="1-D elastic wave", tags=("dt check",))
def bar_wave_speed(E, rho):
    """Longitudinal bar wave speed sqrt(E/rho)."""
    return math.sqrt(E / rho)

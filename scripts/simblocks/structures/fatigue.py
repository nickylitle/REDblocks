"""High-cycle fatigue (Shigley's Mechanical Engineering Design, 10th ed., ch. 6) and Miner damage.
Impact-loaded machines: most parts are overload governed, but shafts, bearing seats and blade hubs see
spin-up/vibration cycling. SI units (MPa where noted, because the Shigley fits are in MPa)."""
import math

from ..registry import block

SRC = "Shigley 10e ch. 6"
SURFACE = {"ground": (1.58, -0.085), "machined": (4.51, -0.265), "cold_drawn": (4.51, -0.265),
           "hot_rolled": (57.7, -0.718), "as_forged": (272.0, -0.995)}     # a, b with Sut in MPa (Table 6-2)


@block("structures", "fatigue", "steel", "endurance_limit", source=SRC + " eq. 6-8")
def endurance_limit_steel(Sut):
    """Rotating-beam endurance limit Se' (Pa): 0.5 Sut for Sut <= 1400 MPa, else 700 MPa."""
    return 0.5 * Sut if Sut <= 1400e6 else 700e6


@block("structures", "fatigue", "general", "marin_surface", source=SRC + " eq. 6-19, Table 6-2")
def marin_surface_factor(Sut, finish="machined"):
    """Surface factor ka = a Sut^b (Sut in MPa inside the fit)."""
    a, b = SURFACE[finish]
    return a * (Sut / 1e6) ** b


@block("structures", "fatigue", "round", "marin_size", source=SRC + " eq. 6-20 (bending/torsion)")
def marin_size_factor(d):
    """Size factor kb for a round in bending/torsion, d in m: 1.24 d_mm^-0.107 (2.79-51 mm),
    1.51 d^-0.157 (51-254 mm), 1 for axial loading (pass d=None)."""
    if d is None:
        return 1.0
    dm = d * 1e3
    if dm <= 2.79:
        return 1.0
    return 1.24 * dm ** -0.107 if dm <= 51 else 1.51 * dm ** -0.157


@block("structures", "fatigue", "general", "basquin_sn", source=SRC + " eqs. 6-13 to 6-16 (f-fraction S-N line)")
def sn_life(sigma_a, Sut, Se, f=None):
    """Cycles to failure on the Shigley S-N line through (1e3, f Sut) and (1e6, Se). f defaults by Sut (Fig. 6-18 fit)."""
    if f is None:
        s = Sut / 6.894757e6 / 1000           # ksi
        f = min(max(1.06 - 2.8e-3 * s + 6.9e-6 * s * s, 0.77), 0.9) if s >= 70 else 0.9
    a = (f * Sut) ** 2 / Se
    b = -math.log10(f * Sut / Se) / 3
    if sigma_a <= Se:
        return math.inf
    return (sigma_a / a) ** (1 / b)


@block("structures", "fatigue", "general", "mean_stress", source=SRC + " Table 6-6 / 6-7")
def fatigue_sf(sigma_a, sigma_m, Se, Sut, Sy=None, criterion="goodman"):
    """Fatigue factor of safety with mean stress: goodman, gerber, asme_elliptic, soderberg."""
    if criterion == "goodman":
        return 1.0 / (sigma_a / Se + sigma_m / Sut)
    if criterion == "soderberg":
        return 1.0 / (sigma_a / Se + sigma_m / Sy)
    if criterion == "gerber":
        if sigma_m == 0:
            return Se / sigma_a
        return 0.5 * (Sut / sigma_m) ** 2 * (sigma_a / Se) * \
            (-1 + math.sqrt(1 + (2 * sigma_m * Se / (Sut * sigma_a)) ** 2))
    if criterion == "asme_elliptic":
        return math.sqrt(1.0 / ((sigma_a / Se) ** 2 + (sigma_m / Sy) ** 2))
    raise ValueError(criterion)


@block("structures", "fatigue", "general", "miner", source="Palmgren-Miner linear damage")
def miner_damage(cycles, lives):
    """Miner sum D = sum n_i / N_i (failure predicted at D >= 1)."""
    return sum(n / N for n, N in zip(cycles, lives) if math.isfinite(N))

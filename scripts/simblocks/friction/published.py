"""Published friction and tire models (sourced replacements for the LEGACY laws in laws.py).

Every block names the paper, equation and access status. "secondary" means the equation was read in a
paper or official doc that quotes the original, not in the original itself.

Recommended use for a small wheel with measured peak / sliding mu (e.g. urethane 2.0 / 1.67):
    brush_mu (Svendenius two-friction brush model) needs mu_s, mu_k and the slip stiffness. Its mu_s is
    the ADHESION limit, not the curve peak, so convert a measured peak with brush_mu_s_for_peak first. magic_formula is the alternative when a full
    mu-slip curve has been measured. Lag the slip with relaxation_lag (distance-based, safe at V = 0).
"""
import math

from ..registry import block
from .._jit import kernel

PACEJKA = ("Pacejka, Tire and Vehicle Dynamics, 2nd ed. 2006 / 3rd ed. 2012; fitting relations as quoted "
           "in arXiv:2502.05479 Eq. 8 and Table III (secondary)")
SVEND = ("Svendenius, Tire Modeling and Friction Estimation, PhD thesis, Lund Univ. 2007, Sec. 4.1 "
         "Eqs. 4.8-4.29; single-mu form Svendenius & Wittenmark, ECC 2003 Eqs. 10-13 (PARTIAL: signs taken "
         "from context; mu_s = mu_k reduction verified analytically)")
STRIBECK = ("Liu et al., Mech. Sci. 6:15-28, 2015, Eq. 6, doi:10.5194/ms-6-15-2015; exponent range 0.5-2 "
            "per Alaci et al., Axioms 13(6):372, 2024, Eq. 10")
LUGRE = ("Canudas de Wit, Olsson, Astrom & Lischinsky, IEEE TAC 40(3):419-425, 1995, "
         "doi:10.1109/9.376053 (secondary: arXiv:2508.00175 Eq. 34)")


# ---------------------------------------------------------------- sliding friction
@block("friction", "coulomb", "sliding", "stribeck_general", kind="kernel", source=STRIBECK,
       tags=("Stribeck", "static friction", "breakaway"))
@kernel
def stribeck_mu_delta(v, mu_s, mu_k, v_s, delta):
    """Stribeck coefficient mu_k + (mu_s - mu_k) exp(-|v/v_s|^delta). delta = 2 is the LuGre standard."""
    return mu_k + (mu_s - mu_k) * math.exp(-abs(v / v_s) ** delta)


@block("friction", "lugre", "sliding", "bristle_step", kind="kernel", source=LUGRE,
       tags=("presliding", "stick-slip", "hysteresis", "dynamic friction"))
@kernel
def lugre_step(z, v, dt, sigma0, sigma1, sigma2, Fc, Fs, vs):
    """One LuGre step. F = s0 z + s1 dz/dt + s2 v, dz/dt = v - s0 |v| z / g(v), g = Fc + (Fs-Fc) e^-(v/vs)^2.
    z is integrated exactly over dt at constant v (stable for any dt). Fc, Fs are forces (mu N).
    Typical: s1 = 2 sqrt(m s0) (critical damping). Returns (F, z_new). F opposes nothing by itself:
    apply -F to the sliding body."""
    g = Fc + (Fs - Fc) * math.exp(-(v / vs) ** 2)
    a = sigma0 * abs(v) / g
    zdot = v - a * z
    if a > 0.0:
        zss = v / a
        z_new = zss + (z - zss) * math.exp(-a * dt)
    else:
        z_new = z + v * dt
    return sigma0 * z + sigma1 * zdot + sigma2 * v, z_new


@block("friction", "coulomb", "sliding", "smooth_stribeck", kind="kernel",
       source="Makkar, Dixon, Sawyer & Hu, ACC 2005 Eq. 2 (continuously differentiable friction)",
       tags=("smooth", "control", "Stribeck"))
@kernel
def makkar_friction(v, g1, g2, g3, g4, g5, g6):
    """g1 (tanh(g2 v) - tanh(g3 v)) + g4 tanh(g5 v) + g6 v. g4 = Coulomb level, g1 + g4 ~ static peak,
    g5 = Coulomb regularization steepness (s/m), g6 = viscous. Returns the friction magnitude law (sign of v)."""
    return g1 * (math.tanh(g2 * v) - math.tanh(g3 * v)) + g4 * math.tanh(g5 * v) + g6 * v


@block("friction", "coulomb", "sliding", "simscape_translational", kind="kernel",
       source=("MathWorks Simscape 'Translational Friction' reference (citing Armstrong & Canudas de Wit, "
               "The Control Handbook, CRC 1995)"),
       tags=("breakaway", "Stribeck", "regularized"))
@kernel
def simscape_friction(v, F_brk, F_c, v_brk, f_visc):
    """sqrt(2e)(F_brk - F_c) exp(-(v/v_St)^2) v/v_St + F_c tanh(v/v_Coul) + f v,
    v_St = v_brk sqrt(2), v_Coul = v_brk / 10. Returns the resisting force (same sign as v)."""
    v_st = v_brk * math.sqrt(2.0)
    v_co = v_brk / 10.0
    return (math.sqrt(2.0 * math.e) * (F_brk - F_c) * math.exp(-(v / v_st) ** 2) * (v / v_st)
            + F_c * math.tanh(v / v_co) + f_visc * v)


# ---------------------------------------------------------------- tires and wheels
@block("friction", "tire", "wheel", "magic_formula", kind="kernel", source=PACEJKA,
       tags=("Pacejka", "mu-slip", "traction"))
@kernel
def magic_formula(x, B, C, D, E):
    """Pacejka pure-slip curve y = D sin(C atan(Bx - E(Bx - atan Bx))). With D = peak mu, y is mu(slip)."""
    bx = B * x
    return D * math.sin(C * math.atan(bx - E * (bx - math.atan(bx))))


@block("friction", "tire", "wheel", "magic_formula_fit", kind="builder", source=PACEJKA,
       tags=("Pacejka", "fit", "mu_s", "mu_k"))
def magic_formula_fit(mu_peak, mu_slide, slip_peak, slip_stiffness=None):
    """B, C, D, E from a measured curve: peak mu, fully sliding mu, slip at peak and initial slope dmu/dslip.
    C = 2 - (2/pi) asin(mu_slide/mu_peak); B = slope/(C D); E = (B xm - tan(pi/2C)) / (B xm - atan(B xm)).
    With no measured slope, slope = 2 mu_peak / slip_peak is used (stand-in from the PSU ME481 notes) and
    flagged. Raises if E >= 1 (curve would have no asymptote)."""
    D = mu_peak
    r = min(max(mu_slide / mu_peak, 0.0), 1.0)
    C = 2.0 - (2.0 / math.pi) * math.asin(r)
    flagged = slip_stiffness is None
    slope = 2.0 * mu_peak / slip_peak if flagged else slip_stiffness
    B = slope / (C * D)
    bx = B * slip_peak
    if C <= 1.0 + 1e-9:            # no peak (mu_slide == mu_peak): E is free, take 0
        E = 0.0
    else:
        E = (bx - math.tan(math.pi / (2.0 * C))) / (bx - math.atan(bx))
    if E >= 1.0:
        raise ValueError(f"E = {E:.3f} >= 1: peak slip too large for this slip stiffness")
    return dict(B=B, C=C, D=D, E=E, slope_assumed=flagged)


@block("friction", "tire", "wheel", "brush_two_mu", kind="kernel", source=SVEND,
       tags=("brush model", "mu_s", "mu_k", "urethane", "traction"))
@kernel
def brush_mu(sigma, C_slip, Fz, mu_s, mu_k):
    """Brush model with parabolic pressure and separate adhesion (mu_s) and sliding (mu_k) friction.
    sigma = |theoretical slip|, C_slip = 2 a^2 c_p (initial slip stiffness, N per unit slip).
    psi = sigma / sigma0, sigma0 = 3 mu_s Fz / C_slip.
    F = C sigma (1 - psi)^2 + mu_k Fz psi^2 (3 - 2 psi) for psi < 1, else mu_k Fz. Returns mu = F / Fz."""
    if Fz <= 0.0:
        return 0.0
    psi = sigma * C_slip / (3.0 * mu_s * Fz)
    if psi >= 1.0:
        return mu_k
    F = C_slip * sigma * (1.0 - psi) ** 2 + mu_k * Fz * psi * psi * (3.0 - 2.0 * psi)
    return F / Fz


@block("friction", "tire", "wheel", "brush_peak_fit", kind="builder", source=SVEND + "; peak solved numerically",
       tags=("fit", "mu_s", "peak"))
def brush_mu_s_for_peak(mu_peak, mu_k):
    """Brush adhesion limit mu_s that makes the brush curve PEAK at a measured mu_peak.
    The brush mu_s is not the curve peak: with mu_s = 2.0, mu_k = 1.67 the curve only peaks near 1.68.
    So a measured breakaway/peak value must be converted with this before it goes into brush_mu.
    Peak of mu(psi) = mu_k + 3(mu_s - mu_k) psi (1-psi)^2 - mu_k (1-psi)^3 over psi in (0, 1)."""
    import numpy as np
    from scipy.optimize import brentq
    psi = np.linspace(1e-4, 1 - 1e-6, 20001)

    def peak(ms):
        return float(np.max(mu_k + 3 * (ms - mu_k) * psi * (1 - psi) ** 2 - mu_k * (1 - psi) ** 3))
    if mu_peak <= mu_k:
        return mu_k
    hi = mu_peak
    while peak(hi) < mu_peak:
        hi *= 2.0
    return brentq(lambda ms: peak(ms) - mu_peak, mu_k, hi, xtol=1e-10)


@block("friction", "tire", "wheel", "brush_force_2d", kind="kernel",
       source=SVEND + "; isotropic brush => force collinear with the slip vector",
       tags=("traction", "drive", "combined slip"))
@kernel
def brush_force_2d(vcx, vcy, w, rw, Fz, C_slip, mu_s, mu_k, vmin):
    """Planar wheel force from contact-patch velocity (vcx, vcy) and spin w, using brush_mu on the slip
    vector magnitude. vmin regularizes the slip denominator at standstill (Pacejka's slip is singular at
    V = 0; pair with relaxation_lag for low-speed accuracy). Returns (Fx, Fy, slip, mu)."""
    sx = vcx - w * rw
    sy = vcy
    sm = math.sqrt(sx * sx + sy * sy)
    vref = max(abs(w * rw), math.sqrt(vcx * vcx + vcy * vcy), vmin)
    sig = sm / vref
    mu = brush_mu(sig, C_slip, Fz, mu_s, mu_k)
    if sm > 1e-12:
        return -mu * Fz * sx / sm, -mu * Fz * sy / sm, sig, mu
    return 0.0, 0.0, 0.0, mu


@block("friction", "tire", "wheel", "relaxation_lag", kind="kernel",
       source=("Romano, Bruzelius, Hjort & Jacobson, Veh. Syst. Dyn. 61(4), 2023, Eq. 13, "
               "doi:10.1080/00423114.2022.2057335; Lee et al., VSD 2017, Eqs. 4-5 (tau = sigma / V)"),
       tags=("transient slip", "relaxation length", "low speed"))
@kernel
def relaxation_lag(y_lag, y_target, V, dt, lam):
    """First-order lag over DISTANCE: dy/ds = (y_target - y)/lam, ds = |V| dt, integrated exactly.
    Holds its value at V = 0 instead of blowing up. lam = slip stiffness / carcass lateral stiffness."""
    return y_target + (y_lag - y_target) * math.exp(-abs(V) * dt / lam)


@block("friction", "rolling", "wheel", "resistance_torque_published", kind="kernel",
       source=("f_R = e / r_dyn, F = f_R N (Hofmann, Logistics Journal 2024, Eqs. 2-3, "
               "doi:10.2195/lj_proc_hofmann_de_202410_01); tanh regularization as the Simscape Coulomb term"),
       tags=("urethane", "rolling resistance", "caster"))
@kernel
def rolling_torque(w, rw, N, f_R, v_eps):
    """Rolling-resistance torque f_R N r tanh(w r / v_eps), opposing spin. Pick v_eps << rolling speed
    (Gonthier rule of thumb: about 0.01 of the characteristic speed)."""
    return f_R * N * rw * math.tanh(w * rw / v_eps)


# ---------------------------------------------------------------- vehicle
@block("rigid_body", "planar", "two_axle", "longitudinal_transfer", kind="analytic",
       source="Gillespie, Fundamentals of Vehicle Dynamics, SAE 1992, Ch. 2 (moment balance; PARTIAL: eq. no. unverified)",
       tags=("load transfer", "wheelie"))
def axle_load_transfer(m, g, L, c_rear, h, ax, grade=0.0):
    """Front and rear axle loads on a grade: Wf = W(c cos th - h sin th)/L - m ax h/L, Wr = W cos th - Wf.
    c_rear = CG distance ahead of the rear axle."""
    W = m * g
    Wf = (W * (c_rear * math.cos(grade) - h * math.sin(grade)) - m * ax * h) / L
    return Wf, W * math.cos(grade) - Wf

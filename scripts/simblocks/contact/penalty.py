"""Penalty contact kernels and rigid-body impact (impulse) blocks."""
import math

from ..registry import block
from .._jit import kernel


@block("contact", "penalty", "general", "normal_force", kind="kernel", source="penalty method",
       tags=("one-sided spring", "unilateral"))
@kernel
def penalty_linear(pen, k):
    """One-sided linear penalty: F = k * pen if pen > 0 else 0."""
    return k * pen if pen > 0.0 else 0.0


@block("contact", "penalty", "general", "normal_force_damped", kind="kernel", source="Kelvin-Voigt, clipped",
       tags=("damping", "restitution"))
@kernel
def penalty_kelvin_voigt(pen, pen_rate, k, c):
    """Linear spring-dashpot contact, clipped so it never pulls: max(k pen + c pen_rate, 0)."""
    if pen <= 0.0:
        return 0.0
    f = k * pen + c * pen_rate
    return f if f > 0.0 else 0.0


@block("contact", "penalty", "general", "normal_force_damped", kind="kernel", source="Hunt & Crossley, J. Appl. Mech. 1975 (via Machado et al. 2012 Eq. 28)",
       tags=("restitution", "nonlinear damping"))
@kernel
def hunt_crossley(pen, pen_rate, K, n, alpha):
    """Hunt-Crossley F = K pen^n (1 + alpha pen_rate), alpha ~ 3(1-e)/(2 v_impact). Zero when not touching."""
    if pen <= 0.0:
        return 0.0
    f = K * pen ** n * (1.0 + alpha * pen_rate)
    return f if f > 0.0 else 0.0


@block("contact", "penalty", "general", "damping_from_restitution",
       source=("Kawaguchi et al., JSME Trans. B 58(551):79-85, 1992, Eq. 11 as given in Itasca PFC 7.0 "
               "'Calibrating the normal critical damping ratio' (position-terminated contact)"),
       tags=("restitution",))
def damping_for_restitution(e, k, m):
    """Dashpot c giving restitution e for linear contact: zeta = -ln e / sqrt(pi^2 + ln^2 e), c = 2 zeta sqrt(k m)."""
    le = math.log(e)
    zeta = -le / math.sqrt(math.pi ** 2 + le * le)
    return 2 * zeta * math.sqrt(k * m)


@block("contact", "impulse", "rigid_body", "effective_mass", source="impulse-momentum, planar",
       tags=("off-center hit", "point mass"))
def effective_mass_at_point(m, I, r_perp):
    """Effective translational mass of a rigid body at a contact point: 1/(1/m + r_perp^2 / I)."""
    inv = (1.0 / m if math.isfinite(m) else 0.0) + (r_perp * r_perp / I if math.isfinite(I) else 0.0)
    return 1.0 / inv


@block("contact", "impulse", "rigid_body", "collision", source="Newton restitution, planar",
       tags=("collision", "energy transfer"))
def collision_impulse(v_rel_normal, m_eff1, m_eff2, e):
    """Normal impulse J = (1+e) v_rel / (1/m1 + 1/m2) and energy lost to the collision."""
    mu = 1.0 / (1.0 / m_eff1 + 1.0 / m_eff2)
    J = (1 + e) * mu * v_rel_normal
    dE = 0.5 * mu * v_rel_normal ** 2 * (1 - e * e)
    return J, dE


@block("contact", "impulse", "spinner", "strike", source="impulse on a spinning rigid blade",
       tags=("spinning blade", "energy transfer", "rotor strike"))
def spinner_strike(I_rotor, omega, R_tip, m_target, e=0.0, I_target=math.inf, r_target=0.0,
                   m_attacker=math.inf):
    """One tangential strike: spinning blade (inertia I at tip radius R) hits a target of mass m_target.
    m_attacker = mass of the frame carrying the bearing-mounted blade (inf = grounded axle);
    chassis rotation is neglected. Returns impulse J, blade speed after, target delta-v, energy absorbed."""
    m_w = I_rotor / (R_tip * R_tip)                        # blade tip effective mass
    m_w = effective_mass_at_point(m_attacker, I_rotor, R_tip) if math.isfinite(m_attacker) else m_w
    m_t = effective_mass_at_point(m_target, I_target, r_target)
    J, dE = collision_impulse(omega * R_tip, m_w, m_t, e)
    w_after = omega - J * R_tip / I_rotor
    return dict(J=J, omega_after=w_after, dv_target=J / m_target, E_absorbed=dE,
                E_before=0.5 * I_rotor * omega ** 2)


MACHADO = ("Machado, Moreira, Flores & Lankarani, Mech. Mach. Theory 2012, Eqs. 28-47, "
           "doi:10.1016/j.mechmachtheory.2012.02.010 (secondary for the originals)")


@block("contact", "penalty", "general", "hysteresis_factor", source=MACHADO,
       tags=("restitution", "Hunt-Crossley", "Lankarani-Nikravesh", "Flores"))
def hysteresis_damping_factor(e, v_impact, model="flores"):
    """alpha for hunt_crossley (F = K d^n (1 + alpha d_dot)) that targets restitution e at impact speed v.
    hunt_crossley: 3(1-e)/(2v), valid for e near 1. lankarani_nikravesh: 3(1-e^2)/(4v), e near 1.
    flores (2011): 8(1-e)/(5 e v), valid from very elastic to very plastic (use for dent-type hits)."""
    if model == "hunt_crossley":
        return 3 * (1 - e) / (2 * v_impact)
    if model == "lankarani_nikravesh":
        return 3 * (1 - e * e) / (4 * v_impact)
    if model == "flores":
        return 8 * (1 - e) / (5 * e * v_impact)
    raise ValueError(model)


@block("contact", "penalty", "general", "restitution_force_terminated",
       source=("Schwager & Poschel, Granular Matter 9:465-469, 2007, as given in Bartz arXiv:2204.10917 "
               "Eqs. 8-11 (secondary)"), tags=("restitution", "clipped contact"))
def restitution_force_terminated(k, m, c):
    """Restitution of a linear spring-dashpot whose force is clipped at zero (no pulling), which ends the
    contact before the spring returns to zero length. Higher than exp(-zeta pi / sqrt(1-zeta^2))."""
    w0 = math.sqrt(k / m)
    beta = c / (2 * m)
    if beta >= w0:
        raise ValueError("overdamped")
    w = math.sqrt(w0 * w0 - beta * beta)
    t = math.acos(2 * beta * beta / (w0 * w0) - 1) / w
    return math.exp(-beta * t)

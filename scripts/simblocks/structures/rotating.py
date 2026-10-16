"""Rotating bodies: rotor and flywheel stress and stored energy.
Roark 7th ed. ch. 16 (rotating disks) and
Timoshenko & Goodier, Theory of Elasticity, sec. 32. Plane stress, uniform thickness."""
import math

from ..registry import block

SRC = "Timoshenko & Goodier sec. 32; Roark 7e ch. 16"


@block("structures", "rotating", "solid_disk", "spin", source=SRC, tags=("rotor", "burst", "flywheel"))
def solid_disk_spin_stress(rho, nu, omega, R):
    """Peak stress in a solid spinning disk: center sigma_r = sigma_t = (3+nu)/8 rho w^2 R^2.
    Returns (sigma_center, sigma_t_rim)."""
    q = rho * omega ** 2 * R * R
    return (3 + nu) / 8 * q, (1 - nu) / 4 * q


@block("structures", "rotating", "annular_disk", "spin", source=SRC, tags=("rotor", "bore", "hub"))
def annular_disk_spin_stress(rho, nu, omega, Ri, Ro):
    """Spinning annular disk with free bore and rim: peak hoop stress at the bore
    sigma_t(Ri) = (3+nu)/4 rho w^2 (Ro^2 + (1-nu)/(3+nu) Ri^2)."""
    return (3 + nu) / 4 * rho * omega ** 2 * (Ro * Ro + (1 - nu) / (3 + nu) * Ri * Ri)


@block("structures", "rotating", "annular_disk", "spin_field", source=SRC)
def annular_disk_spin_field(rho, nu, omega, Ri, Ro, r):
    """Radial and hoop stress at radius r in a spinning annular disk (free edges). Returns (sr, st)."""
    q = rho * omega ** 2
    A = (3 + nu) / 8 * q
    sr = A * (Ri * Ri + Ro * Ro - Ri * Ri * Ro * Ro / (r * r) - r * r)
    st = A * (Ri * Ri + Ro * Ro + Ri * Ri * Ro * Ro / (r * r) - (1 + 3 * nu) / (3 + nu) * r * r)
    return sr, st


@block("structures", "rotating", "thin_ring", "spin", source="hoop equilibrium", tags=("rim", "drum"))
def thin_ring_spin_stress(rho, omega, R):
    """Thin rotating ring hoop stress sigma = rho w^2 R^2 (= rho v^2)."""
    return rho * (omega * R) ** 2


@block("structures", "rotating", "bar", "spin", source="radial equilibrium of a uniform bar",
       tags=("rotating bar", "blade root", "tip mass"))
def bar_spin_root_stress(rho, omega, L, A=None, m_tip=0.0, r_hub=0.0):
    """Root tensile stress of a uniform bar of length L spinning about one end (or a hub at r_hub):
    sigma = rho w^2 (L^2/2 + r_hub L) + m_tip w^2 (r_hub + L) / A."""
    s = rho * omega ** 2 * (L * L / 2 + r_hub * L)
    if m_tip and A:
        s += m_tip * omega ** 2 * (r_hub + L) / A
    return s


@block("structures", "rotating", "general", "energy", source="KE = I w^2 / 2", tags=("stored energy", "flywheel"))
def spin_energy(I, omega):
    """Stored rotational kinetic energy."""
    return 0.5 * I * omega * omega


@block("structures", "rotating", "general", "tip_speed", source="v = w R")
def tip_speed(rpm, R):
    """Tip speed (m/s) from rpm and radius."""
    return rpm * 2 * math.pi / 60 * R


@block("structures", "rotating", "general", "burst_speed", source="sigma_peak = su", tags=("overspeed",))
def burst_rpm_solid_disk(rho, nu, su, R):
    """RPM at which the solid-disk center stress reaches su (no defects, ductile: an upper bound)."""
    w = math.sqrt(8 * su / ((3 + nu) * rho * R * R))
    return w * 60 / (2 * math.pi)


@block("structures", "rotating", "general", "bite", source="bite = v_close * T_rev / n_teeth",
       tags=("bite depth", "tooth"))
def bite_depth(v_close, rpm, n_teeth=1):
    """Bite (m): distance closed per tooth pass."""
    return v_close * 60.0 / (rpm * n_teeth)


@block("structures", "rotating", "general", "gyro", source="tau = I w x Omega", tags=("gyroscopic", "turn"))
def gyro_torque(I_spin, omega_spin, omega_precession):
    """Gyroscopic torque magnitude from turning a spinning rotor at rate Omega."""
    return I_spin * omega_spin * omega_precession


_SPIN_SHAPES = ("solid_disk", "annular_disk", "thin_ring", "bar_root")


@block("structures", "rotating", "general", "spin", source=SRC + "; hoop and radial equilibrium",
       tags=("rotor", "flywheel", "burst", "bore", "hub", "rim", "drum", "blade root", "overspeed"))
def spin_stress(shape, omega=None, rpm=None, rho=None, nu=None, su=None, material=None, R=None, Ri=None, Ro=None,
                L=None, A=None, m_tip=0.0, r_hub=0.0, r=None):
    """Centrifugal stress in a spinning body, and the burst speed when the strength su is known.

    shape
      solid_disk    R: centre stress (3+nu)/8 rho w^2 R^2 (= sigma_max) and rim hoop stress sigma_t_rim
      annular_disk  Ri, Ro, free bore and rim: peak hoop stress at the bore (sigma_max); with r also the radial and
                    hoop stress at radius r (sr, st)
      thin_ring     R: hoop stress rho w^2 R^2
      bar_root      L (bar spinning about one end, or about a hub at r_hub), optional tip mass m_tip on area A:
                    root tensile stress
    omega (rad/s) or rpm: spin speed. rho, nu, su: density, Poisson ratio, ultimate strength; material
    (catalog name) supplies any of them that are not given.
    Returns dict(sigma_max, ...) as listed; with su also burst_rpm, the speed at which sigma_max reaches su
    (solid disk: closed form; other shapes: stress scales with w^2). Without a speed only burst_rpm is returned.
    """
    from ..materials.lookup import mat_props, need
    sh = str(shape).strip().lower()
    if sh not in _SPIN_SHAPES:
        raise ValueError(f"spin_stress: unknown shape {shape!r}; use one of {', '.join(_SPIN_SHAPES)}")
    m = mat_props(material, rho=rho, nu=nu, su=su)
    need("rho (or material)", m.rho, "spin_stress")
    if omega is None and rpm is not None:
        omega = rpm * 2 * math.pi / 60
    w = 1.0 if omega is None else omega
    if sh == "solid_disk":
        sc, st = solid_disk_spin_stress(m.rho, need("nu (or material)", m.nu, "spin_stress"), w,
                                        need("R", R, "spin_stress solid_disk"))
        out = dict(sigma_max=sc, sigma_center=sc, sigma_t_rim=st)
    elif sh == "annular_disk":
        nu_ = need("nu (or material)", m.nu, "spin_stress")
        Ri_, Ro_ = need("Ri", Ri, "spin_stress annular_disk"), need("Ro", Ro, "spin_stress annular_disk")
        out = dict(sigma_max=annular_disk_spin_stress(m.rho, nu_, w, Ri_, Ro_))
        if r is not None:
            out["sr"], out["st"] = annular_disk_spin_field(m.rho, nu_, w, Ri_, Ro_, r)
    elif sh == "thin_ring":
        out = dict(sigma_max=thin_ring_spin_stress(m.rho, w, need("R", R, "spin_stress thin_ring")))
    else:
        out = dict(sigma_max=bar_spin_root_stress(m.rho, w, need("L", L, "spin_stress bar_root"), A, m_tip, r_hub))
    if omega is None:
        out = {}
    if not math.isnan(m.su):
        if sh == "solid_disk":
            out["burst_rpm"] = burst_rpm_solid_disk(m.rho, m.nu, m.su, R)
        else:
            s1 = (out["sigma_max"] / (omega * omega)) if omega else \
                {"annular_disk": lambda: annular_disk_spin_stress(m.rho, m.nu, 1.0, Ri, Ro),
                 "thin_ring": lambda: thin_ring_spin_stress(m.rho, 1.0, R),
                 "bar_root": lambda: bar_spin_root_stress(m.rho, 1.0, L, A, m_tip, r_hub)}[sh]()
            out["burst_rpm"] = math.sqrt(m.su / s1) * 60 / (2 * math.pi)
    elif omega is None:
        raise ValueError("spin_stress needs omega or rpm (or su for burst_rpm only)")
    return out

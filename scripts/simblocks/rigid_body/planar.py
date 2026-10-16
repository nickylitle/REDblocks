"""Planar rigid-body kinematics and force accumulation kernels (njit-safe)."""
import math

from ..registry import block
from .._jit import kernel


@block("rigid_body", "planar", "frames", "world_to_body", kind="kernel", source="2-D rotation")
@kernel
def world_to_body(vx, vy, psi):
    """Rotate a world vector into the body frame at heading psi. Returns (bx, by)."""
    c, s = math.cos(psi), math.sin(psi)
    return c * vx + s * vy, -s * vx + c * vy


@block("rigid_body", "planar", "frames", "body_to_world", kind="kernel", source="2-D rotation")
@kernel
def body_to_world(bx, by, psi):
    """Rotate a body-frame vector into the world frame. Returns (vx, vy)."""
    c, s = math.cos(psi), math.sin(psi)
    return c * bx - s * by, s * bx + c * by


@block("rigid_body", "planar", "point", "velocity", kind="kernel", source="v_P = v_O + w x r",
       tags=("contact patch", "wheel"))
@kernel
def point_velocity(vbx, vby, r, px, py):
    """Body-frame velocity of a point at body coords (px, py) on a body with yaw rate r. Returns (vx, vy)."""
    return vbx - r * py, vby + r * px


@block("rigid_body", "planar", "point", "force_moment", kind="kernel", source="M_z = x Fy - y Fx")
@kernel
def force_moment_about_origin(Fx, Fy, px, py):
    """Yaw moment of a force (Fx, Fy) applied at (px, py)."""
    return px * Fy - py * Fx


@block("rigid_body", "planar", "body", "accelerations", kind="kernel", source="Newton-Euler, planar")
@kernel
def planar_accel(Fx_body, Fy_body, Mz, m, Izz, psi):
    """World-frame translational accelerations and yaw acceleration. Returns (ax, ay, alpha)."""
    c, s = math.cos(psi), math.sin(psi)
    return (c * Fx_body - s * Fy_body) / m, (s * Fx_body + c * Fy_body) / m, Mz / Izz


@block("rigid_body", "planar", "three_point_stance", "normal_loads", kind="kernel",
       source="Statics: Gillespie 1992 Ch. 2 longitudinal moment balance; lateral transfer m a_y h / T per side (PARTIAL: eq. nos. unverified). Lag ax with relaxation_lag, not a fixed tau",
       tags=("load transfer", "wheelie", "nose load"))
@kernel
def three_point_normal_loads(m, g, xc, L, T, h, ax_lag, ay_lag):
    """Two wheels on an axle (track T) plus one skid at L ahead of the axle; COM xc ahead of axle at height h.
    Returns (N_left, N_right, N_skid, lift_flag)."""
    NB = m * (g * xc - h * ax_lag) / L
    lift = 0.0
    if NB < 0.0:
        NB = 0.0
        lift = 1.0
    Nw = m * g - NB
    dN = -2.0 * h * m * ay_lag / T
    NL = 0.5 * (Nw + dN)
    NR = 0.5 * (Nw - dN)
    if NL < 0.0:
        NL = 0.0
        lift = 2.0
    if NR < 0.0:
        NR = 0.0
        lift = 2.0
    return NL, NR, NB, lift


@block("rigid_body", "planar", "body", "ICR", kind="kernel", source="v = w x r_ICR", tags=("pivot point",))
@kernel
def instant_center(vbx, vby, r):
    """Instantaneous center of rotation in body coordinates (nan if not rotating)."""
    if abs(r) < 1e-9:
        return math.nan, math.nan
    return -vby / r, vbx / r

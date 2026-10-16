"""Friction and tire force laws (njit-safe kernels). First extracted from a user-supplied vehicle drive sim and generalized.
Sourced replacements live in published.py; the two LEGACY tire curves here are kept only to reproduce old runs.
Example measured values (friction.data): urethane 20A on dusty painted plywood mu_k 1.67 / mu_s 2.0; steel button head skid mu_k 0.203."""
import math

from ..registry import block
from .._jit import kernel


@block("friction", "coulomb", "sliding", "regularized", kind="kernel", source="Coulomb tanh regularization = Simscape Translational Friction Coulomb term (v_eps = v_brk/10); Marques, Flores & Lankarani 2016 Eqs. 17-20",
       tags=("skid", "stick-slip"))
@kernel
def coulomb_tanh(v, N, mu, v_eps):
    """Smooth Coulomb friction F = -mu N tanh(v / v_eps). Pick v_eps << typical sliding speed."""
    return -mu * N * math.tanh(v / v_eps)


@block("friction", "coulomb", "sliding", "stribeck", kind="kernel",
       source="Liu et al. 2015 Eq. 6 / LuGre steady-state g(v) with exponent 2 (Canudas de Wit 1995)", tags=("skid", "bolt", "static friction"))
@kernel
def stribeck_mu(v, mu_s, mu_k, v_s):
    """Stribeck friction coefficient mu_k + (mu_s - mu_k) exp(-(v/v_s)^2)."""
    return mu_k + (mu_s - mu_k) * math.exp(-(v / v_s) ** 2)


@block("friction", "coulomb", "sliding", "planar_skid", kind="kernel",
       source="Stribeck coefficient (Liu 2015 Eq. 6, exponent 2) on the sliding speed; direction regularized as Gonthier dir(v, eps) (Marques 2016 Eq. 19)", tags=("2-D", "skid"))
@kernel
def skid_force_2d(vx, vy, N, mu_s, mu_k, v_s, eps):
    """2-D sliding-contact friction force opposing velocity (vx, vy) with Stribeck coefficient. Returns (Fx, Fy)."""
    vm = math.sqrt(vx * vx + vy * vy)
    mu = mu_k + (mu_s - mu_k) * math.exp(-(vm / v_s) ** 2)
    den = math.sqrt(vm * vm + eps * eps)
    return -mu * N * vx / den, -mu * N * vy / den


@block("friction", "tire", "wheel", "combined_slip_mu", kind="kernel",
       source="LEGACY: user-supplied drive sim (no published basis found; use brush_mu or magic_formula)", tags=("traction", "urethane"))
@kernel
def tire_mu(sig, mu_s, mu_k, sig_p):
    """Combined-slip friction curve: linear at small slip, peak ~mu_s at sig_p, decays to mu_k."""
    sig0 = 0.5 * sig_p
    return mu_k * math.tanh(sig / sig0) + (mu_s - mu_k) * (sig / sig_p) * math.exp(1.0 - sig / sig_p)


@block("friction", "tire", "wheel", "combined_slip_force", kind="kernel",
       source="LEGACY: user-supplied drive sim (no published basis found; use brush_force_2d)", tags=("traction", "drive"))
@kernel
def tire_force_2d(vcx, vcy, w, rw, N, mu_s, mu_k, sig_p, vmin):
    """Planar wheel contact force from contact-patch velocity (vcx, vcy) and wheel spin w.
    Returns (Fx, Fy, slip_speed, mu)."""
    sx = vcx - w * rw
    sy = vcy
    sm = math.sqrt(sx * sx + sy * sy)
    vref = max(abs(w * rw), math.sqrt(vcx * vcx + vcy * vcy), vmin)
    sig = sm / vref
    sig0 = 0.5 * sig_p
    mu = mu_k * math.tanh(sig / sig0) + (mu_s - mu_k) * (sig / sig_p) * math.exp(1.0 - sig / sig_p)
    if sm > 1e-12:
        return -mu * N * sx / sm, -mu * N * sy / sm, sm, mu
    return 0.0, 0.0, 0.0, mu


@block("friction", "rolling", "wheel", "resistance_torque", kind="kernel",
       source="Same form as rolling_torque (Hofmann 2024 Eqs. 2-3 + Simscape tanh); kept for the legacy argument order", tags=("urethane", "rolling resistance"))
@kernel
def rolling_resistance_torque(w, rw, N, Crr, v_eps):
    """Rolling resistance torque Crr N rw tanh(w rw / v_eps)."""
    return Crr * N * rw * math.tanh(w * rw / v_eps)


@block("friction", "bearing", "shaft", "drag_torque", kind="kernel", source="SKF simplified model",
       tags=("rotor bearing", "spin-down"))
@kernel
def bearing_drag_torque(w, F_radial, mu_b, d_bore, M0):
    """Bearing friction torque M0 + 0.5 mu F d (SKF simplified), signed against w."""
    m = M0 + 0.5 * mu_b * F_radial * d_bore
    return -m * math.tanh(w / 1.0)

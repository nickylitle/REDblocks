"""Adaptive time step for explicit sims (stiffness- and gap-driven, never deformation-driven).

Rule (standard explicit-dynamics practice: critical step from the stiffest active element, Belytschko,
Liu & Moran 'Nonlinear Finite Elements for Continua and Structures' ch. 6; contact look-ahead so the step
is already fine at the instant of first touch):

    dt_stab   = C * min over ACTIVE elements of sqrt(m_eff / k_elastic)    (elastic k even while yielding,
                                                                           because unloading is elastic)
    approach  : if a gap is closing, dt_target = clamp(gap / (N * v_close), dt_stab, dt_max)
    in contact: dt_target = dt_stab
    free      : dt_target = dt_max
    smoothing : dt never grows by more than `growth` per step; it shrinks immediately
    caps      : dt in [dt_min, dt_max] always

Verification convention: an adaptive sim's run(params, dt) treats `dt` as a refinement scale. dt_min,
dt_max and C all scale by dt / dt_ref, so verify_all's dt-halving still measures convergence.
Semi-implicit Euler is only symplectic at constant dt, so the energy ledger is the watchdog here.
"""
import math

from ..registry import block
from .._jit import kernel


@block("integrators", "adaptive", "contact", "step_size", kind="kernel",
       source="critical-step rule (Belytschko, Liu & Moran ch. 6) + contact look-ahead",
       tags=("variable dt", "adaptive", "multi-scale"))
@kernel
def adaptive_dt(dt_prev, dt_stab, in_contact, gap, v_close, dt_min, dt_max, n_lookahead, growth):
    """Next step size. gap >= 0 is the smallest open gap, v_close > 0 its closing speed (<= 0 if opening).
    dt_stab is the stability-based step for the stiffest element that is or is about to be active."""
    if in_contact:
        target = dt_stab
    elif v_close > 0.0:
        target = gap / (n_lookahead * v_close)
        if target < dt_stab:
            target = dt_stab
        if target > dt_max:
            target = dt_max
    else:
        target = dt_max
    grown = dt_prev * growth
    dt = target if target < dt_prev else (grown if grown < target else target)
    if dt < dt_min:
        dt = dt_min
    if dt > dt_max:
        dt = dt_max
    return dt


@block("integrators", "adaptive", "contact", "stable_step", kind="kernel",
       source="dt_crit = 2/w_max (central difference); C is the safety fraction",
       tags=("variable dt", "stability"))
@kernel
def stable_step(k_elastic, m_eff, C):
    """Stability-based step C * sqrt(m_eff / k) (C ~ 0.1 for accurate penalty contact)."""
    return C * math.sqrt(m_eff / k_elastic)

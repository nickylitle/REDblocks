"""DC / BLDC (DC-equivalent) motors, gearboxes, batteries, ESC behavior.
First extracted from a user-supplied vehicle drive sim (motor current, gearbox loss direction, bus fixed point) and generalized."""
import math

from ..registry import block
from .._jit import kernel


@block("actuators", "motor", "dc", "from_datasheet", kind="builder", source="DC motor equations",
       tags=("Kv", "brushless", "brushed"))
def motor_from_kv(kv_rpm_per_V, R_phase, I_limit=math.inf, J_rotor=0.0, I0=0.0, name="motor"):
    """DC-equivalent motor from Kv (rpm/V), winding resistance R (ohm) and ESC current limit.
    Kt = 60 / (2 pi Kv) (N.m/A = V.s/rad). I0 = no-load current (friction torque Kt I0)."""
    Kt = 60.0 / (2 * math.pi * kv_rpm_per_V)
    return dict(name=name, Kt=Kt, R=R_phase, Imax=I_limit, J=J_rotor, I0=I0)


@block("actuators", "motor", "dc", "from_curve", kind="builder",
       source="DC motor equations fit to the vendor 12 V curve (fit method from a user-supplied drive sim)", tags=("brushed", "gearmotor"))
def motor_from_curve(V_test, rpm_free_out, stall_torque_out, stall_current, G):
    """Fit Kt, R and gearbox efficiency to a gearmotor's free speed, stall torque, stall current at V_test."""
    w_free_m = rpm_free_out * G * 2 * math.pi / 60
    Ke = V_test / w_free_m
    R = V_test / stall_current
    eta = stall_torque_out / (G * Ke * stall_current)
    return dict(Kt=Ke, R=R, eta=eta, G=G, Imax=math.inf)


@block("actuators", "motor", "dc", "current", kind="kernel", source="DC motor equations; ESC coast/brake logic LEGACY: user-supplied drive sim",
       tags=("ESC", "current limit", "drag brake", "coast"))
@kernel
def motor_current(u, w_motor, vbus, Kt, R, Imax, brake):
    """Winding current for throttle u in [-1, 1] at motor speed w (rad/s). brake=0: open circuit near
    zero throttle (coast); brake=1: drag brake. Current clipped to +/- Imax (ESC limit)."""
    if brake < 0.5 and abs(u) < 0.02:
        return 0.0
    I = (u * vbus - Kt * w_motor) / R
    if I > Imax:
        return Imax
    if I < -Imax:
        return -Imax
    return I


@block("actuators", "gearbox", "general", "output_torque", kind="kernel",
       source="LEGACY: user-supplied drive sim; sets eta_b = eta_f, published data says eta_b < eta_f; use gearbox_output_torque_dir", tags=("efficiency", "back-drive"))
@kernel
def gearbox_output_torque(T_motor, w_motor, G, eta):
    """Output torque with efficiency applied in the correct direction: motoring T G eta, back-driven T G / eta."""
    if T_motor * w_motor >= 0.0:
        return eta * G * T_motor
    return G * T_motor / eta


@block("actuators", "battery", "pack", "bus_voltage", kind="kernel", source="Plett 2015 Ch. 2 equivalent-circuit model, R0-only case (Thevenin); closed-form constant-power root",
       tags=("sag", "internal resistance"))
@kernel
def bus_voltage(V_oc, R_int, P_load, v_guess):
    """Bus voltage for a constant-power load P on a source V_oc with internal resistance R:
    solves V = V_oc - R P / V (stable root), starting from v_guess."""
    disc = V_oc * V_oc - 4.0 * R_int * P_load
    if disc <= 0.0:
        return 0.5 * V_oc          # beyond max power transfer: collapse to the limit
    return 0.5 * (V_oc + math.sqrt(disc))


@block("actuators", "battery", "pack", "open_circuit", source="LiPo typical OCV curve (approx)",
       tags=("state of charge", "LiPo"))
def lipo_ocv(soc, cells, v_full=4.2, v_empty=3.3, v_nom=3.75):
    """Approximate LiPo pack open-circuit voltage vs state of charge (piecewise; fine for sag studies)."""
    import numpy as np
    s = np.array([0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0])
    v = np.array([v_empty, 3.55, 3.68, v_nom, 3.95, 4.1, v_full])
    return cells * float(np.interp(soc, s, v))


@block("actuators", "esc", "current_limited", "supply", kind="kernel",
       source="DC motor + averaged PWM ESC (lossless, duty <= 1) on a Thevenin source; algebraic, no lag",
       tags=("ESC", "current limit", "sag", "duty"))
@kernel
def esc_supply(Veff, R0, E, Rm, Imax):
    """Full-throttle ESC between a source (Veff behind R0) and a motor (back-EMF E, winding Rm).
    duty 1 if the motor cannot reach Imax: I = (Veff - E)/(R0 + Rm), battery current = I.
    otherwise duty < 1 holds I = Imax and the source supplies P = (E + Imax Rm) Imax at the stable root.
    Returns (I_motor, I_batt, v_bus, duty). Consistent energy: v_bus I_batt = (E + I Rm) I."""
    if Veff <= E:
        return 0.0, 0.0, Veff, 1.0
    I1 = (Veff - E) / (R0 + Rm)
    if I1 <= Imax:
        return I1, I1, Veff - R0 * I1, 1.0
    P = (E + Imax * Rm) * Imax
    if R0 <= 0.0:
        return Imax, P / Veff, Veff, (E + Imax * Rm) / Veff
    disc = Veff * Veff - 4.0 * R0 * P
    if disc < 0.0:
        disc = 0.0
    Ib = (Veff - math.sqrt(disc)) / (2.0 * R0)
    v = Veff - R0 * Ib
    return Imax, Ib, v, (E + Imax * Rm) / v


@block("actuators", "esc", "throttled", "supply", kind="kernel",
       source="averaged PWM ESC with its duty capped at the throttle, on a Thevenin source; energy-consistent",
       tags=("ESC", "throttle", "current limit", "sag", "duty"))
@kernel
def esc_supply_thr(Veff, R0, E, Rm, Imax, thr):
    """ESC whose duty is at most thr (throttle 0..1). Below the current limit the motor sees thr times the bus:
    I = (thr Veff - E) / (Rm + thr^2 R0), battery current Ib = thr I. Above it, the ESC holds Imax (duty < thr).
    Returns (I_motor, I_batt, v_bus, duty), with v_bus I_batt = (E + I Rm) I exactly (no energy made or lost)."""
    if thr >= 1.0:
        return esc_supply(Veff, R0, E, Rm, Imax)
    if thr <= 0.0 or thr * Veff <= E:
        return 0.0, 0.0, Veff, 0.0
    I1 = (thr * Veff - E) / (Rm + thr * thr * R0)
    if I1 <= Imax:
        Ib = thr * I1
        return I1, Ib, Veff - R0 * Ib, thr
    return esc_supply(Veff, R0, E, Rm, Imax)


@block("actuators", "battery", "pack", "open_circuit_kernel", kind="kernel", source="LiPo typical OCV curve (approx)",
       tags=("state of charge", "LiPo", "njit"))
@kernel
def lipo_ocv_k(soc, cells):
    """njit-safe twin of lipo_ocv (same table, nominal 3.75 V, 3.3-4.2 V per cell) for use inside loops."""
    s0, s1, s2, s3, s4, s5 = 0.0, 0.05, 0.2, 0.5, 0.8, 0.95
    if soc <= 0.0:
        return cells * 3.3
    if soc >= 1.0:
        return cells * 4.2
    if soc < s1:
        v = 3.3 + (3.55 - 3.3) * (soc - s0) / (s1 - s0)
    elif soc < s2:
        v = 3.55 + (3.68 - 3.55) * (soc - s1) / (s2 - s1)
    elif soc < s3:
        v = 3.68 + (3.75 - 3.68) * (soc - s2) / (s3 - s2)
    elif soc < s4:
        v = 3.75 + (3.95 - 3.75) * (soc - s3) / (s4 - s3)
    elif soc < s5:
        v = 3.95 + (4.1 - 3.95) * (soc - s4) / (s5 - s4)
    else:
        v = 4.1 + (4.2 - 4.1) * (soc - s5) / (1.0 - s5)
    return cells * v


@block("actuators", "motor", "rotor", "spin_up_current_limited", source="constant torque then back-EMF limited",
       tags=("spin-up time", "flywheel", "rotor"))
def spin_up_time(J_total, Kt, R, V, I_limit, w_target, G=1.0):
    """Time to reach w_target (load rad/s) for a current-limited motor, gear ratio G (motor:load).
    Phase 1: I = I_limit (constant torque) until back-EMF allows less; phase 2: exponential approach."""
    Jm = J_total / (G * G)            # reflect load inertia to motor shaft
    wm_target = w_target * G
    w_free = V / Kt
    if wm_target >= w_free:
        return math.inf
    w_knee = (V - I_limit * R) / Kt   # motor speed where the current limit releases
    tau = Jm * R / (Kt * Kt)
    if wm_target <= w_knee:
        return Jm * wm_target / (Kt * I_limit)
    t1 = Jm * max(w_knee, 0.0) / (Kt * I_limit)
    return t1 + tau * math.log((w_free - max(w_knee, 0.0)) / (w_free - wm_target))


@block("actuators", "motor", "dc", "copper_loss", kind="kernel", source="I^2 R")
@kernel
def copper_loss(I, R):
    """Winding Joule loss I^2 R (W)."""
    return I * I * R


@block("actuators", "gearbox", "general", "backdrive_efficiency",
       source=("Matsuki, Nagano & Fujimoto, IEEE/ASME T-Mech 24(6), 2019, Eq. 38, as reproduced in Sim & Ramos "
               "arXiv:2011.02506 Eq. 15 (secondary)"), tags=("back-drive", "efficiency", "planetary"))
def backdrive_efficiency(eta_f, G):
    """Back-driven efficiency from forward efficiency for a reduction G (motor turns : output turns, e.g. 20).
    eta_b = (2 eta_f - 1 + g^2)/((1 - g^2) eta_f + 2 g^2), g = 1/G; 0 (self-locking) if eta_f <= (1-g^2)/2.
    Always eta_b <= eta_f."""
    g = 1.0 / G if G > 1.0 else G
    if eta_f <= (1 - g * g) / 2:
        return 0.0
    return (2 * eta_f - 1 + g * g) / ((1 - g * g) * eta_f + 2 * g * g)


@block("actuators", "gearbox", "general", "output_torque_directional", kind="kernel",
       source=("Sim & Ramos, arXiv:2106.01842 Eqs. 2, 5, 6 (directional efficiency: forward eta_f, "
               "backward eta_b < eta_f)"), tags=("efficiency", "back-drive"))
@kernel
def gearbox_output_torque_dir(T_motor, w_motor, G, eta_f, eta_b):
    """Output torque with direction-dependent efficiency: motoring G T eta_f; back-driven (load drives the
    motor) the load sees G T / eta_b. Get eta_b from backdrive_efficiency if not measured."""
    if T_motor * w_motor >= 0.0:
        return eta_f * G * T_motor
    if eta_b <= 0.0:
        return 1e30 if T_motor > 0.0 else -1e30   # self-locking: caller must treat as locked
    return G * T_motor / eta_b


@block("actuators", "battery", "cell", "rc_ecm_step", kind="kernel",
       source=("Plett, Battery Management Systems Vol. I, Artech House 2015, Ch. 2 (OCV + R0 + R1||C1); "
               "equations as in arXiv:2406.08626 Eqs. 1-3 (secondary)"), tags=("sag", "recovery", "LiPo"))
@kernel
def ecm_constant_power(OCV, R0, R1, C1, iR1, P_load, dt):
    """One step of the one-RC equivalent-circuit model under a constant-power load.
    Solves P = v i with v = OCV - R1 iR1 - R0 i, then advances iR1 exactly over dt.
    Returns (v, i, iR1_new). R1 = 0 reduces to bus_voltage (Thevenin)."""
    Veff = OCV - R1 * iR1
    disc = Veff * Veff - 4.0 * R0 * P_load
    if disc < 0.0:
        disc = 0.0            # beyond max power transfer: collapse to the limit
    if R0 > 0.0:
        i = (Veff - math.sqrt(disc)) / (2.0 * R0)
    else:
        i = P_load / Veff
    v = Veff - R0 * i
    if R1 > 0.0 and C1 > 0.0:
        a = math.exp(-dt / (R1 * C1))
        iR1_new = a * iR1 + (1.0 - a) * i
    else:
        iR1_new = i
    return v, i, iR1_new

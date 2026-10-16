"""Battery life of Flapjack over a 180 s fight: weapon spin-ups, weapon hold, normal driving and pushing on one 6S pack.

Weapon: two 2807 1350 KV drone motors, each with its own current-limited ESC, friction-driving the blade (motor can
on a tread ring on the blade hub, ratio G motor turns per blade turn). Blade inertia J, windage (strip theory with a
low effective Cd: the air co-rotates with a bar spinning in its own plane) and a bearing drag torque. Each motor has
a no-load current I0 (iron + bearing losses). Spin-up = ESC at its current limit until the blade reaches the target
speed; then a governor holds speed. A hit drops the blade to a fraction of the target speed.

Drive: normal driving replays the electrical power of the legacy drive sim's aggressive 30 s segment at 6S
(drive_power_trace_6S.npz). Pushing: robot blocked; both wheels lumped on one spin DOF, two drive motors at full
throttle through the gearbox, tire force = Coulomb friction at the push traction limit (urethane slipping up to
cleats locked into the floor, which stalls the motors at their current limit).

Pack: 6 cells in series, one-RC equivalent circuit (OCV(SOC) + R0 + R1||C1) under the summed constant-power load.
Cell resistance scales with capacity (same C-rate cells): R0 = r450 * 0.45 Ah / C.

Failure (user, proposed and accepted): the pack is too small if the fight uses more than 80 % of rated capacity,
or the loaded voltage drops under 3.3 V per cell at any time.
"""
import math
import os
import sys

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from simblocks.actuators.motors import esc_supply, ecm_constant_power, lipo_ocv_k, motor_current
from simblocks.aero.basic import bar_windage_torque
from simblocks.friction.laws import coulomb_tanh

_TR = np.load(os.path.join(HERE, "drive_power_trace_6S.npz"))
DRIVE_T, DRIVE_P = np.ascontiguousarray(_TR["t"]), np.ascontiguousarray(_TR["P"])
DRIVE_DT = float(DRIVE_T[1] - DRIVE_T[0])

DEFAULTS = dict(
    # pack
    cap_mAh=450.0, cells=6, r450_mohm=15.0, R1_frac=0.5, tau_rc=20.0, usable=0.80, v_cut=3.3,
    # weapon
    kv=1350.0, R_w=0.052, I0_w=1.5, n_wmot=2, I_lim_w=40.0, G_w=1.37, eta_fd=0.85,
    J_blade=8.5e-4, rpm_target=12000.0, Cd_wind=0.2, R_blade=0.095, t_blade=0.00635, T_bearing=0.003,
    # weapon use
    n_spinups=8, f_after_hit=0.4,
    # drive
    duty_drive=0.5, push_s=20.0, push_len=3.0, F_push=20.0, G_d=22.6, kv_d=2300.0, R_d=0.15, I_lim_d=20.0,
    eta_d=0.82, J_wheel=6e-6, J_rot_d=8e-7, r_wheel=0.022,
    # misc
    P_quiescent=3.0, t_fight=180.0, seed=0,
    # switches for knockouts
    windage_on=1.0, hold_on=1.0,
)


def schedule(p):
    """Event script for one fight, deterministic from p['seed']: weapon re-spin times and drive mode per 0.5 s slot
    (0 idle, 1 normal driving, 2 pushing)."""
    rng = np.random.default_rng(int(p["seed"]) + 12345)
    T = p["t_fight"]; slot = 0.5; ns = int(round(T / slot))
    mode = np.zeros(ns, np.int64)
    n_push = int(round(p["push_s"] / p["push_len"]))
    plen = max(1, int(round(p["push_len"] / slot)))
    free = list(range(0, ns - plen))
    placed = 0
    while placed < n_push and free:
        s = int(rng.choice(free))
        if mode[s:s + plen].any():
            free.remove(s); continue
        mode[s:s + plen] = 2; placed += 1
        free = [f for f in free if f + plen <= s or f >= s + plen]
    # normal driving fills the given fraction of the non-push time in 1.5 s chunks
    rest = np.nonzero(mode == 0)[0]
    want = int(round(p["duty_drive"] * len(rest)))
    chunks = [rest[i:i + 3] for i in range(0, len(rest), 3)]
    order = rng.permutation(len(chunks))
    got = 0
    for k in order:
        if got >= want:
            break
        mode[chunks[k]] = 1; got += len(chunks[k])
    n_hit = max(0, int(round(p["n_spinups"])) - 1)
    hits = np.sort(rng.uniform(5.0, T - 5.0, n_hit)) if n_hit else np.zeros(0)
    return slot, mode, hits


@njit(cache=True)
def _fight(dt, T, slot, mode, hits, cap_Ah, cells, R0, R1, C1, v_cut,
           Kt_w, R_w, I0_w, n_wmot, I_lim_w, G_w, eta_fd, J_b, w_t, Cd, R_b, t_b, T_brg, f_hit, windage_on, hold_on,
           Kt_d, R_d, I_lim_d, G_d, eta_d, J_w, r_w, F_push, drive_t, drive_p, drive_dt, P_q):
    n = int(round(T / dt))
    wb = 0.0; ww = 0.0; soc = 1.0; iR1 = 0.0; vbus = lipo_ocv_k(1.0, cells)
    Ah = 0.0; vmin = 1e9; Ipk = 0.0
    E_cell = 0.0; E_Rloss = 0.0
    E_w_in = 0.0; E_w_cu = 0.0; E_w_nl = 0.0; E_fd = 0.0; E_wind = 0.0; E_brg = 0.0; E_hit = 0.0
    E_d_in = 0.0; E_push_in = 0.0; E_push_cu = 0.0; E_slip = 0.0; E_q = 0.0
    KE0 = 0.0
    t_spin = 0.0; n_done = 0; spinning = 1
    hi = 0; nh = hits.shape[0]
    t_first_full = -1.0
    for k in range(n):
        t = k * dt
        # ---------------- hits
        if hi < nh and t >= hits[hi]:
            KEb = 0.5 * J_b * wb * wb
            wb = f_hit * w_t if wb > f_hit * w_t else wb
            spinning = 1                                           # a hit starts a new spin-up
            E_hit += KEb - 0.5 * J_b * wb * wb
            hi += 1
        # ---------------- weapon
        wm = G_w * wb
        E = Kt_w * wm
        Tw = windage_on * bar_windage_torque(wb, R_b, t_b, 1.2, Cd) + T_brg
        if spinning == 1:
            # spin-up: ESC at its current limit until the blade reaches the target (one spin-up per hit)
            I, Ib_, v_, d_ = esc_supply(vbus, 0.0, E, R_w, I_lim_w)
            t_spin += dt
        else:
            # governor: just enough motor current to hold the target speed (hold_on = 0: throttle off, coast)
            I = hold_on * (Tw / (G_w * eta_fd) / n_wmot / Kt_w + I0_w)
            if I > I_lim_w:
                I = I_lim_w
            if hold_on > 0.0 and wb < 0.995 * w_t:
                # fell below target (current limit or sag): spin back up at full current
                spinning = 1
                I, Ib_, v_, d_ = esc_supply(vbus, 0.0, E, R_w, I_lim_w)
        P_w = n_wmot * (E + I * R_w) * I
        if I > 0.0:
            Tm = Kt_w * (I - I0_w)                                 # shaft torque per motor
            Tb = n_wmot * Tm * G_w * eta_fd
        else:
            Tm = 0.0                                               # ESC off: no-load losses drag the blade
            Tb = -n_wmot * Kt_w * I0_w * G_w if wb > 0.0 else 0.0
        acc = (Tb - Tw) / J_b
        wb_new = wb + acc * dt
        if wb_new < 0.0:
            wb_new = 0.0
        if spinning == 1 and wb_new >= w_t:
            wb_new = w_t
            spinning = 0
            n_done += 1
            if t_first_full < 0.0:
                t_first_full = t
        # ledger pieces (power x dt), weapon
        E_w_in += P_w * dt
        E_w_cu += n_wmot * I * I * R_w * dt
        E_w_nl += n_wmot * Kt_w * I0_w * wm * dt if (I > 0.0 or wb > 0.0) else 0.0
        E_fd += n_wmot * Tm * wm * (1.0 - eta_fd) * dt
        E_wind += (Tw - T_brg) * wb * dt
        E_brg += T_brg * wb * dt
        # ---------------- drive
        s = int(t / slot)
        if s >= mode.shape[0]:
            s = mode.shape[0] - 1
        md = mode[s]
        P_d = 0.0
        if md == 1:
            j = int((t % 30.0) / drive_dt)
            if j >= drive_p.shape[0]:
                j = drive_p.shape[0] - 1
            P_d = drive_p[j]
            E_d_in += P_d * dt
            ww = 0.0
        elif md == 2:
            Id = motor_current(1.0, G_d * ww, vbus, Kt_d, R_d, I_lim_d, 1.0)
            Ed = Kt_d * G_d * ww
            P_d = 2.0 * (Ed + Id * R_d) * Id
            Fx = -coulomb_tanh(ww * r_w, F_push, 1.0, 0.02)             # friction force on the tread, opposing slip
            Tw_d = 2.0 * Kt_d * Id * G_d * eta_d - Fx * r_w
            E_push_in += P_d * dt
            E_push_cu += 2.0 * Id * Id * R_d * dt
            E_slip += Fx * ww * r_w * dt
            E_fd += 2.0 * Kt_d * Id * G_d * ww * (1.0 - eta_d) * dt
            ww = ww + Tw_d / J_w * dt
            if ww < 0.0:
                ww = 0.0
        else:
            ww = 0.0
        # ---------------- pack
        P = P_w + P_d + P_q
        E_q += P_q * dt
        ocv = lipo_ocv_k(soc, cells)
        v, i, iR1n = ecm_constant_power(ocv, R0, R1, C1, iR1, P, dt)
        E_cell += ocv * i * dt
        E_Rloss += (R0 * i * i + R1 * iR1 * iR1) * dt
        iR1 = iR1n
        soc -= i * dt / (cap_Ah * 3600.0)
        Ah += i * dt / 3600.0
        if v < vmin:
            vmin = v
        if i > Ipk:
            Ipk = i
        vbus = v
        wb = wb_new
    KE1 = 0.5 * J_b * wb * wb
    return (Ah, vmin, Ipk, soc, E_cell, E_Rloss, E_w_in, E_w_cu, E_w_nl, E_fd, E_wind, E_brg, E_hit, E_d_in,
            E_push_in, E_push_cu, E_slip, E_q, KE1 - KE0, t_spin, float(n_done), t_first_full)


KEYS = ["Ah", "v_min", "I_peak", "soc_end", "E_cell", "E_Rloss", "E_weapon_in", "E_weapon_cu", "E_weapon_noload",
        "E_transmission", "E_windage", "E_bearing", "E_hits", "E_drive_in", "E_push_in", "E_push_cu", "E_tire_slip",
        "E_quiescent", "dKE_blade", "t_spinning", "n_full", "t_first_full"]


def run(params=None, dt=1e-4):
    p = dict(DEFAULTS); p.update(params or {})
    slot, mode, hits = schedule(p)
    C = p["cap_mAh"] / 1000.0
    R0c = p["r450_mohm"] * 1e-3 * 0.45 / C
    R0 = p["cells"] * R0c; R1 = p["R1_frac"] * R0; C1 = p["tau_rc"] / R1 if R1 > 0 else 0.0
    Kt_w = 60.0 / (2 * math.pi * p["kv"]); Kt_d = 60.0 / (2 * math.pi * p["kv_d"])
    J_w = 2.0 * (p["J_wheel"] + p["J_rot_d"] * p["G_d"] ** 2)
    w_t = p["rpm_target"] * 2 * math.pi / 60
    r = _fight(dt, p["t_fight"], slot, mode, hits, C, p["cells"], R0, R1, C1, p["v_cut"],
               Kt_w, p["R_w"], p["I0_w"], p["n_wmot"], p["I_lim_w"], p["G_w"], p["eta_fd"], p["J_blade"], w_t,
               p["Cd_wind"], p["R_blade"], p["t_blade"], p["T_bearing"], p["f_after_hit"], p["windage_on"],
               p["hold_on"], Kt_d, p["R_d"], p["I_lim_d"], p["G_d"], p["eta_d"], J_w, p["r_wheel"], p["F_push"],
               DRIVE_T, DRIVE_P, DRIVE_DT, p["P_quiescent"])
    o = dict(zip(KEYS, [float(x) for x in r]))
    o["used_frac"] = o["Ah"] / C
    o["v_min_cell"] = o["v_min"] / p["cells"]
    o["Wh"] = o["E_cell"] / 3600.0
    o["C_peak"] = o["I_peak"] / C
    o["mAh_used"] = o["Ah"] * 1000.0
    # failure margins (user definition): capacity and sag, both SF = allowed / demand (>1 passes)
    o["SF_capacity"] = p["usable"] / max(o["used_frac"], 1e-9)
    o["SF_sag"] = o["v_min_cell"] / p["v_cut"] if p["v_cut"] > 0 else math.inf
    o["passes"] = 1.0 if (o["SF_capacity"] >= 1.0 and o["SF_sag"] >= 1.0) else 0.0
    # energy ledger: cell energy = resistive loss + every sink + blade KE left at the end
    sinks = (o["E_Rloss"] + o["E_weapon_cu"] + o["E_weapon_noload"] + o["E_transmission"] + o["E_windage"]
             + o["E_bearing"] + o["E_hits"] + o["E_drive_in"] + o["E_push_cu"] + o["E_tire_slip"] + o["E_quiescent"]
             + o["dKE_blade"])
    o["_ledger_rel"] = abs(o["E_cell"] - sinks) / max(o["E_cell"], 1e-9)
    o["_sinks"] = sinks
    o["_mode_push_s"] = float((mode == 2).sum() * slot); o["_mode_drive_s"] = float((mode == 1).sum() * slot)
    o["_n_hits"] = float(len(hits))
    return o


def min_capacity(params, lo=100.0, hi=3000.0, tol=5.0, dt=1e-4):
    """Smallest rated capacity (mAh) that passes both failure modes, by bisection (pass/fail flips once in
    capacity: a bigger pack of the same cells has lower resistance and more charge)."""
    def ok(c):
        return run({**params, "cap_mAh": c}, dt)["passes"] > 0.5
    if not ok(hi):
        return float("nan"), None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    o = run({**params, "cap_mAh": hi}, dt)
    o["governs"] = "capacity" if o["SF_capacity"] < o["SF_sag"] else "sag"
    return hi, o


if __name__ == "__main__":
    import time
    t0 = time.time(); o = run(); print(f"first {time.time()-t0:.1f}s")
    t0 = time.time(); o = run(); print(f"second {time.time()-t0:.2f}s")
    for k, v in o.items():
        print(f"{k:18s} {v:.5g}")
    t0 = time.time(); c, oc = min_capacity({}); print("min cap", c, oc["governs"], f"{time.time()-t0:.1f}s")

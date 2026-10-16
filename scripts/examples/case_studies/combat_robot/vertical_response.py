"""What does each tip load do to Flapjack's vertical momentum, with and without the wheels on the floor?

Each load table is applied at the tip of Flapjack's (rigid) blade with nothing else in the arena:
  - free body: no floor, no gravity (the load's own momentum, no ground reaction)
  - wheels, compliant: the sim's supports (k = 3e5 N/m per contact, TPU wheel + urethane tire assumption)
  - wheels, stiff: k = 3e6 N/m (floor reacts inside the 0.5 ms pulse)
An upward tip load pitches the robot nose-up about its centre of mass, which drives the rear wheels into the
floor; the floor pushes back up, adding vertical momentum and cutting the rotation. Closed-form check: impulsive
pivot about the rear wheel contact.
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from spinner_vs_spinner import run, build, DEFAULTS
from simblocks.diagnostics.loads import half_sine_train
from simblocks.rigid_body.spatial import quat_to_matrix
from simblocks.viz import theme as T
from simblocks.viz.diagram import Schematic

OUT = os.path.join(HERE, "out", "vertical_response"); os.makedirs(OUT, exist_ok=True)
LOADDIR = os.path.join(HERE, "out", "rigid_blade_tip_load")
T_LOAD = 1.0e-3                     # load starts after 1 ms of settling on the supports
T_END = 0.030
STATIC = dict(H_rpm=1e-3)          # blade not spinning, load stays at the front: the MATLAB rigid-blade setup
CASES = {"static blade, free body": dict(gravity=0.0, z_lift=1.0, **STATIC),
         "static blade, wheels k 3e5": dict(STATIC),
         "static blade, wheels k 3e6": dict(k_sup=3.0e6, **STATIC),
         "spinning blade, wheels k 3e5": {}}


def loads():
    tu, Fu = half_sine_train([3408, 3408], [0.53e-3, 0.53e-3], [0.0, 1.06e-3], 5e-3, n_per=40)
    L = {"current load (2 half-sines, Z)": (tu, np.c_[0 * Fu, 0 * Fu, Fu])}
    for name, f in (("worst out-of-plane (sim)", "worst_out_of_plane_tip_load_table.txt"),
                    ("worst overall (sim)", "worst_overall_tip_load_table.txt")):
        d = np.loadtxt(os.path.join(LOADDIR, f), comments="%")
        L[name] = (d[:, 0], d[:, 1:4])
    return L


def pitch_deg(row):
    """Nose elevation: angle of the robot's forward axis above the horizontal (yaw-independent, 90 = nose straight up)."""
    R = quat_to_matrix(np.ascontiguousarray(row[9:13]))
    return math.degrees(math.atan2(R[2, 0], math.hypot(R[0, 0], R[1, 0])))


def analytic(Jz, b0):
    """Impulsive estimates for an upward impulse Jz at the blade tip (blade pointing straight ahead)."""
    H = b0["bots"][0]
    m = H["m"]; Iyy = H["J"][1, 1] + 2 * H["mt"] * H["R"] ** 2 / 2
    x_com = H["cb"][0]                                     # body COM in the robot frame (robot COM at 0)
    a = H["d"][0] + H["R"]                                  # tip ahead of the robot COM
    xw = -0.04                                              # rear wheel contact
    free_v = Jz / m; free_w = Jz * a / Iyy
    v_rear = free_v - free_w * (0 - xw)                     # rear wheel vertical velocity, free body
    I_P = Iyy + m * xw ** 2
    w_p = Jz * (a - xw) / I_P
    v_p = w_p * (0 - xw)
    return dict(free_v=free_v, free_w=free_w, v_rear_free=v_rear, pivot_v=v_p, pivot_w=w_p,
                pivot_floor_J=m * v_p - Jz, m=m, Iyy=Iyy, a=a, xw=xw)


def main():
    T.apply()
    b0 = build(dict(DEFAULTS))
    m_tot = b0["bots"][0]["m"]
    w = DEFAULTS["H_rpm"] * 2 * math.pi / 60
    phase = -math.degrees(w * T_LOAD) % 360                  # arm 0 points straight ahead when the load starts
    res = {}
    for ln, (tl, Fl) in loads().items():
        Jz = float(np.sum(0.5 * (Fl[1:, 2] + Fl[:-1, 2]) * np.diff(tl)))
        Jy = float(np.sum(0.5 * (Fl[1:, 1] + Fl[:-1, 1]) * np.diff(tl)))
        for cn, ov in CASES.items():
            p = dict(gap0=5.0, flexH=0.0, phase_H_deg=0.0 if "static" in cn else phase, t_end=T_END, t_after=0.0, hist_dt=5e-6,
                     ext_load=dict(t=tl + T_LOAD, F=Fl, arm=0), **ov)
            o, ex = run(p, history=True)
            H = ex["hist"]; t = H[:, 0] - T_LOAD
            pz = m_tot * H[:, 27]
            pitch = np.array([pitch_deg(r) for r in H])
            z = H[:, 8] - H[0, 8]
            res[(ln, cn)] = dict(t=t, pz=pz, pitch=pitch, z=z, Jz=Jz, Jy=Jy, led=o["_ledger_rel"],
                                 vlat=np.hypot(H[:, 25], H[:, 26]))
            k = np.searchsorted(t, 0.020)
            hop = z[k] + max(H[k, 27], 0) ** 2 / (2 * 9.81) if "free" not in cn else float("nan")
            res[(ln, cn)]["hop"] = hop
            print(f"{ln:34s} | {cn:26s} | Jz {Jz:+.2f}  pz@2ms {pz[np.searchsorted(t, 0.002)]:+.2f}  pz@20ms "
                  f"{pz[k]:+.2f} N s | pitch@20ms {pitch[k]:+.1f} deg | hop {hop*1e3:.0f} mm | lateral v "
                  f"{res[(ln, cn)]['vlat'][k]:.2f} m/s | ledger {o['_ledger_rel']*100:.2f}%")
    an = {ln: analytic(res[(ln, list(CASES)[0])]["Jz"], b0) for ln in loads()}
    import json
    json.dump({f"{ln} | {cn}": dict(Jz=r["Jz"], pz_2ms=float(r["pz"][np.searchsorted(r["t"], 0.002)]),
                                     pz_20ms=float(r["pz"][np.searchsorted(r["t"], 0.020)]),
                                     pitch_20ms=float(r["pitch"][np.searchsorted(r["t"], 0.020)]), hop_mm=r["hop"] * 1e3)
               for (ln, cn), r in res.items()}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for ln, a in an.items():
        print(f"analytic {ln}: free v {a['free_v']:.2f} m/s, w {a['free_w']:.0f} rad/s, rear wheel {a['v_rear_free']:+.2f} m/s"
              f" (negative = into the floor); pivot: v {a['pivot_v']:.2f} m/s, w {a['pivot_w']:.0f} rad/s, floor adds "
              f"{a['pivot_floor_J']:+.2f} N s")

    # ---------------- figure
    L = list(loads())
    fig = plt.figure(figsize=(13, 7.6))
    top = T.header(fig, "Vertical momentum from each tip load",
                   "Load applied at Flapjack's blade tip, blade pointing ahead, nothing else in the arena",
                   tag="SIMBLOCKS // FLOOR REACTION")
    gs = fig.add_gridspec(2, 3, left=0.065, right=0.985, bottom=0.17, top=top - 0.06, wspace=0.28, hspace=0.5)
    cols = {c: (T.SERIES[i] if i < 3 else T.INK_3) for i, c in enumerate(CASES)}
    for j, ln in enumerate(L):
        a1 = fig.add_subplot(gs[0, j]); a2 = fig.add_subplot(gs[1, j])
        for cn in CASES:
            r = res[(ln, cn)]
            ls = (0, (4, 2)) if "spinning" in cn else "-"
            a1.plot(r["t"] * 1e3, r["pz"], color=cols[cn], lw=1.5, ls=ls, label=cn)
            a2.plot(r["t"] * 1e3, r["pitch"], color=cols[cn], lw=1.5, ls=ls, label=cn)
        a1.axhline(res[(ln, list(CASES)[0])]["Jz"], color=T.STATUS["WARN"], lw=1.0, ls=(0, (1, 2)), label="load Z impulse")
        a1.set_title(ln, fontsize=9.5); a1.set_xlabel("time after load start (ms)"); a2.set_xlabel("time after load start (ms)")
        if j == 0:
            a1.set_ylabel("vertical momentum (N s)"); a2.set_ylabel("nose elevation (deg)")
        a1.set_xlim(-0.5, T_END * 1e3 - 1); a2.set_xlim(-0.5, T_END * 1e3 - 1)
        T.style_ticks(a1); T.style_ticks(a2)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.045), frameon=False)
    T.footer(fig, "Vertical momentum = robot mass x COM vertical velocity (gravity on except the free body). Rigid blade; "
                  "supports: two rear wheels (mu 0.05 rolling, 1.6 lateral) and the front skid.")
    figs = [T.save(fig, os.path.join(OUT, "vertical_momentum"))]

    sch = Schematic("Floor reaction to a tip load",
                    "Upward tip load -> nose-up pitch -> rear wheels pushed into the floor -> floor pushes up")
    sch.body("tip", "point", "")
    sch.body("blade", "rotor", "Rigid blade", info=[f"tip {an[L[0]]['a']*1e3:.0f} mm ahead of COM"])
    sch.body("bot", "mass", "Flapjack", info=[f"m = {m_tot:.3f} kg", f"I_pitch = {an[L[0]]['Iyy']*1e3:.2f} g m^2"])
    sch.body("floor", "ground", "")
    sch.link("tip", "blade", "rigid", "Tip load", "table (t, X, Y, Z)", "rigid_blade_tip_load")
    sch.link("blade", "bot", "rigid", "Bearing", "spin DOF", "gyrostat_accel")
    sch.link("bot", "floor", "spring", "Rear wheels + skid", "k 3e5 or 3e6 N/m\none-sided", "supports")
    sch.probe("bot", "vertical momentum, pitch")
    figs.append(sch.draw(os.path.join(OUT, "floor_schematic")))
    return res, an, figs


if __name__ == "__main__":
    main()

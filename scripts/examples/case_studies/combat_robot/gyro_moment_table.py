"""Force + moment tables at the blade centre for a NON-spinning rigid-blade FE model, with the spinning blade's
gyroscopic effect added as a moment.

Why: a spinning blade with angular momentum h (along its axis) pushes on its hub with M_gyro = -w x h whenever the
robot body rotates at w. A static rigid blade in Ansys cannot produce that term, so it is supplied as an applied
moment at the blade centre, on top of the tip-load moment r x F.

How: the Z-only worst-case tip load is applied (body-fixed, blade pointing forward) to Flapjack alone on its wheels
with the blade spinning; the body rate w(t) is taken from that run and M_gyro(t) computed from it. Open loop: the
Ansys model's own rotation will differ a little from the sim's; for a stiff mount the difference is small.
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
from spinner_vs_spinner import run, DEFAULTS
from simblocks.viz import theme as T

OUT = os.path.join(HERE, "out", "rigid_blade_tip_load")
SRC = os.path.join(OUT, "ANSYS_Z_only_force_at_tip.txt")
R_TIP = 0.095
T_LOAD = 1.0e-3
T_TABLE = 5.0e-3                     # same end time as the user's tables
T_SIM = 0.030


def body_to_user(v):
    """sim body axes (x forward, y left, z up) -> user axes (X right, Y forward, Z up)."""
    return np.c_[-v[:, 1], v[:, 0], v[:, 2]]


def case(gyro_on=1.0, spin=True):
    d = np.loadtxt(SRC, comments="%")
    tl, Fz = d[:, 0], d[:, 3]
    F = np.c_[0 * Fz, 0 * Fz, Fz]                         # arm frame = body frame at the front (Z up)
    p = dict(gap0=5.0, flexH=0.0, t_end=T_SIM, t_after=0.0, hist_dt=2.5e-6, gyro_on=gyro_on,
             ext_load=dict(t=tl + T_LOAD, F=F, arm=0, on_body=1))
    if not spin:
        p["H_rpm"] = 1e-3
    o, ex = run(p, history=True)
    H = ex["hist"]; t = H[:, 0] - T_LOAD
    w_body = H[:, 71:74]
    spin_abs = H[:, 4]                                    # blade spin about its axis (rad/s)
    h = DEFAULTS["H_Is"] * spin_abs                        # blade angular momentum (N m s), along body z
    hv = np.c_[0 * h, 0 * h, h]
    Mg_body = -np.cross(w_body, hv)                        # moment of the spinning rotor ON the hub/frame
    return dict(t=t, w=w_body, h=h, Mg=body_to_user(Mg_body), led=o["_ledger_rel"], tl=tl, Fz=Fz)


def main():
    T.apply()
    g = case(1.0, True)
    k = case(0.0, True)              # knockout: same spin, gyroscopic term removed
    print(f"blade h at load start {g['h'][np.searchsorted(g['t'], 0)]:.3f} N m s; ledger {g['led']*100:.2f}%")
    for tm in (0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3):
        i = np.searchsorted(g["t"], tm)
        print(f"t {tm*1e3:5.1f} ms  w_body (deg/s) {np.degrees(g['w'][i]).round(0)}  M_gyro user (N m) {g['Mg'][i].round(1)}"
              f"  | gyro off w {np.degrees(k['w'][i]).round(0)}")
    # independent check of the formula at one instant: |M| = h * |w_perp|
    i = np.searchsorted(g["t"], 2e-3)
    print("check |M| vs h*|w_xy|:", round(float(np.linalg.norm(g['Mg'][i])), 3),
          round(float(g['h'][i] * np.hypot(g['w'][i, 0], g['w'][i, 1])), 3))

    # ---- tables: union of the load table's times and a 0.05 ms grid to 5 ms
    tl, Fz = g["tl"], g["Fz"]
    tg = np.unique(np.r_[tl[tl <= T_TABLE], np.arange(0, T_TABLE + 1e-12, 5e-5), T_TABLE])
    Fz_g = np.interp(tg, tl, Fz)
    Mg = np.c_[[np.interp(tg, g["t"], g["Mg"][:, c]) for c in range(3)]].T
    Mtip = np.c_[R_TIP * Fz_g, 0 * Fz_g, 0 * Fz_g]          # r x F, r = (0, R, 0) user, F = (0, 0, Fz)
    M = Mtip + Mg
    J = float(np.sum(0.5 * (Fz_g[1:] + Fz_g[:-1]) * np.diff(tg)))
    hdr = ["Worst vertical (Z-only) hit, applied at the BLADE CENTRE with the moment table below (same rows).",
           "Frame: X side to side (+ right), Y forward/back (+ forward), Z up. Blade (static in FE) pointing forward.",
           f"Z impulse {J:+.3f} N s. Moment = tip load x {R_TIP*1e3:.0f} mm  +  gyroscopic moment of the spinning blade "
           f"(h = {g['h'][np.searchsorted(g['t'], 0)]:.2f} N m s, counterclockwise seen from above), from the sim's body rate."]
    with open(os.path.join(OUT, "ANSYS_Z_gyro_force_at_centre.txt"), "w") as fh:
        for x in hdr:
            fh.write(f"% {x}\n")
        fh.write("% time_s\tFx_N\tFy_N\tFz_N\n")
        for a, b in zip(tg, Fz_g):
            fh.write(f"{a:.6g}\t0\t0\t{b:.6g}\n")
    with open(os.path.join(OUT, "ANSYS_Z_gyro_moment_at_centre.txt"), "w") as fh:
        for x in hdr:
            fh.write(f"% {x}\n")
        fh.write("% time_s\tMx_Nm\tMy_Nm\tMz_Nm\n")
        for row in zip(tg, *M.T):
            fh.write("\t".join(f"{v:.6g}" for v in row) + "\n")
    with open(os.path.join(OUT, "ANSYS_Z_gyro_moment_only_30ms.txt"), "w") as fh:
        fh.write("% Gyroscopic moment alone (no tip-load part), to 30 ms: it lasts as long as the robot keeps rotating.\n")
        fh.write("% time_s\tMx_Nm\tMy_Nm\tMz_Nm\n")
        tt = np.arange(0, T_SIM - T_LOAD, 2.5e-4)
        for a in tt:
            v = [np.interp(a, g["t"], g["Mg"][:, c]) for c in range(3)]
            fh.write(f"{a:.6g}\t{v[0]:.6g}\t{v[1]:.6g}\t{v[2]:.6g}\n")

    # ---- figure
    fig = plt.figure(figsize=(13, 5.6))
    top = T.header(fig, "Moment at the blade centre with gyroscopic effect",
                   "Z-only worst-case tip load, Flapjack on its wheels, blade spinning; gyro moment = -w x h on the hub",
                   tag="SIMBLOCKS // LOAD EXPORT")
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, bottom=0.2, top=top - 0.07, wspace=0.25)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1])
    a1.plot(tg * 1e3, Mtip[:, 0], color=T.INK_3, lw=1.4, ls=(0, (4, 3)), label="Mx from tip load only")
    a1.plot(tg * 1e3, M[:, 0], color=T.SERIES[0], lw=1.6, label="Mx total (with gyro)")
    a1.plot(tg * 1e3, M[:, 1], color=T.SERIES[1], lw=1.6, label="My (gyro only)")
    a1.set_xlabel("time after load start (ms)"); a1.set_ylabel("moment at blade centre (N m)")
    a1.set_title("Exported moment table (0-5 ms)"); a1.legend(loc="upper right", fontsize=7.5); T.style_ticks(a1)
    tt = g["t"]; m = (tt >= 0) & (tt <= 0.028)
    a2.plot(tt[m] * 1e3, g["Mg"][m, 0], color=T.SERIES[0], lw=1.5, label="gyro Mx")
    a2.plot(tt[m] * 1e3, g["Mg"][m, 1], color=T.SERIES[1], lw=1.5, label="gyro My")
    a2.plot(tt[m] * 1e3, g["Mg"][m, 2], color=T.SERIES[2], lw=1.5, label="gyro Mz")
    a2.axvline(T_TABLE * 1e3, color=T.STATUS["WARN"], lw=1.0, ls=(0, (4, 3)))
    a2.set_xlabel("time after load start (ms)"); a2.set_ylabel("gyroscopic moment (N m)")
    a2.set_title("Gyro moment keeps going while the robot rotates", pad=12)
    a2.legend(loc="upper right", fontsize=7.5); T.style_ticks(a2)
    T.footer(fig, "Gyro moment from the sim's body rate (open loop). h = blade spin inertia x spin rate. The dashed line "
                  "marks the 5 ms table end; a 30 ms gyro-only table is included.")
    T.save(fig, os.path.join(OUT, "gyro_moment"))
    return tg, Fz_g, M


if __name__ == "__main__":
    main()

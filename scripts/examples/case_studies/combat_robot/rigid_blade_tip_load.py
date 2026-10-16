"""Worst-case effective TIP LOAD for a structural model that treats Flapjack's blade as infinitely rigid.

Source: spinner_vs_spinner.py. Each flexible arm of the H blade pushes on the rigid rest of the blade through
its tooth spring, applied at the tip: that force IS the effective tip load a rigid-blade model should see. It
already contains the cap set by the blade bending (the arm yields), the real rise/fall shape, and any second hit.

Steps
1. Blade strength scaled so the flap (out-of-plane) first-yield tip load equals the user's bending load (3408 N).
2. Re-run the study's engagements (both blades bending) with the tip load recorded.
3. Rank by impulse delivered to the blade (lag + flap), pick the worst, verify that run, export a table in the
   user's format (time, X, Y, Z), arm frame: X radial (steady centrifugal pull removed), Y in-plane along the
   direction of rotation, Z out-of-plane (flap, positive up).

  python rigid_blade_tip_load.py            # uses the samples in out/spinner_vs_spinner/mc.pkl
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from spinner_vs_spinner import run, build, DEFAULTS, make_spec
from simblocks.diagnostics.loads import pulse_metrics, half_sine_train, fe_load_table, write_load_table, split_pulses
from simblocks.diagnostics.verify import verify_all
from simblocks.viz import theme as T
from simblocks.viz.diagram import Schematic
from simblocks.viz.report import results_report

OUT = os.path.join(HERE, "out", "rigid_blade_tip_load"); os.makedirs(OUT, exist_ok=True)
F_USER = 3408.0                     # peak of the user's current half-sine load (= AR600 first yield at 95 mm)
F_BEND = float(os.environ.get("TIP_CAP_N", 6110.0))   # most the blade passes out-of-plane: AR600 full plastic at the
                                                       # centre (95 mm) with the x1.2 impact-rate factor (worst case)
R_TIP = 0.095                       # blade centre to tip (moment arm for the centre tables)
USER = dict(peaks=[F_USER, F_USER], durations=[0.53e-3, 0.53e-3], starts=[0.0, 1.06e-3], t_end=5e-3)
HIST_DT = 2.5e-6
WINDOW = (-0.1e-3, 6e-3)            # exported window around the first blade load


def strength_scale():
    """Scale the blade's bending strength so the MOST the arm can pass out-of-plane (full plastic collapse incl.
    hardening, the top of the hinge table) is F_BEND: the user's 'tip load required to bend the blade'."""
    b = build(dict(DEFAULTS))
    fc = float(b["bots"][0]["spring_info"]["flap"]["F_tab"].max())
    return F_BEND / fc, fc


def tip_load(hist, b):
    """Tip load on the rigid H blade from each H arm, arm frame (radial out, along rotation, flap). The radial
    channel has the quasi-static centrifugal pull m_t R w(t)^2 removed at every instant (w = current spin)."""
    t = hist[:, 0]; w = hist[:, 4]
    out = []
    for j in range(2):                       # arms 0, 1 belong to the H blade
        s = float(b["A_sign"][j])
        R, L, Z = s * hist[:, 59 + 3 * j], s * hist[:, 60 + 3 * j], hist[:, 61 + 3 * j]
        cf = float(b["A_m"][j]) * float(b["A_R"][j]) * w ** 2
        out.append((R - cf, L, Z))
    return t, out


def _imp(t, y):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def extract(hist, b):
    """Per H arm: the tip load while the blade is actually being hit. Window = first blade load (-0.1 ms) to the end
    of the last blade contact plus the arm unloading (until the bending load falls under 5% of its peak, at most
    half a flap period). Free ringing after that is blade vibration, not load, and a rigid model cannot carry it.
    X (radial) is averaged over one axial-mode period: that mode (~20 kHz) carries no net momentum."""
    t, arms = tip_load(hist, b)
    contact = (hist[:, 1] + hist[:, 2]) > 0.0                     # blade-blade or blade-body contact
    bots = b["bots"]
    m_t = float(b["A_m"][0])
    T_ax = 2 * math.pi / math.sqrt(float(b["A_kax"][0]) / m_t)
    T_flap = 2 * math.pi / math.sqrt(float(b["A_kf"][0]) / m_t)
    nav = max(1, int(round(T_ax / (t[1] - t[0]))))
    recs = []
    for j, (X, Y, Z) in enumerate(arms):
        F = np.hypot(Y, Z)
        if not contact.any() or F.max() <= 0:
            recs.append(dict(arm=j, J_lat=0.0, J_x=0.0, J_y=0.0, J_z=0.0, F_pk=0.0, Fz_pk=0.0, Fy_pk=0.0, n_pulses=0,
                             t=t[:2] - t[0], R=np.zeros(2), L=np.zeros(2), Z=np.zeros(2)))
            continue
        act = np.nonzero(F > 0.05 * F.max())[0]
        t0 = t[act[0]] - 1e-4
        t_last = t[np.nonzero(contact)[0][-1]]
        # end where the load has come back to zero: first zero crossing of the dominant bending component after
        # the last contact (or the bending load under 5% of its peak), at most one flap period later
        # each bending component ends at its own first zero crossing after the last contact (at most one flap
        # period later) and is held at zero after that; the window ends when the last of them has
        m0 = (t >= t0) & (t <= t_last)
        Y = Y.copy(); Z = Z.copy(); ends = []
        for comp in (Y, Z):
            sgn = np.sign(_imp(t[m0], comp[m0])) or 1.0
            after = np.nonzero((t > t_last) & (sgn * comp <= 0.0))[0]
            te = min(t[after[0]] if after.size else t[-1], t_last + T_flap)
            comp[t > te] = 0.0
            ends.append(te)
        t1 = max(ends)
        m = (t >= t0) & (t <= t1 + 1e-12)
        Xs = np.convolve(X, np.ones(nav) / nav, mode="same")
        tt = t[m]
        Jx, Jy, Jz = _imp(tt, Xs[m]), _imp(tt, Y[m]), _imp(tt, Z[m])
        recs.append(dict(arm=j, J_lat=math.hypot(Jy, Jz), J_x=Jx, J_y=Jy, J_z=Jz, F_pk=float(F[m].max()),
                         Fz_pk=float(np.abs(Z[m]).max()), Fy_pk=float(np.abs(Y[m]).max()),
                         n_pulses=len(split_pulses(tt, F[m], frac=0.05, min_gap=2e-5)),
                         t=tt - tt[0], R=Xs[m], L=Y[m], Z=Z[m]))
    return recs


def _one(a):
    i, params = a
    o, ex = run(params, history=True)
    recs = extract(ex["hist"], ex["build"])
    best = max(recs, key=lambda r: r["J_lat"])               # bending components: in-plane + out-of-plane
    bz = max(recs, key=lambda r: abs(r["J_z"]))
    return dict(i=i, params=params, out=o, all=best, z=bz)


def tip_spec_run(p, dt):
    """Run + the exported tip-load quantities, for verify_all (the checks then cover what is exported)."""
    o, ex = run({**p, "hist_dt": HIST_DT}, dt=dt, history=True)
    best = max(extract(ex["hist"], ex["build"]), key=lambda r: r["J_lat"])
    bz = max(extract(ex["hist"], ex["build"]), key=lambda r: abs(r["J_z"]))
    return {**o, "tip_J_lat": best["J_lat"], "tip_J_x": best["J_x"], "tip_J_z": bz["J_z"],
            "tip_F_lat_pk": best["F_pk"], "tip_Fz_pk": bz["Fz_pk"]}


def tip_spec(params, unc, z_case=False):
    from simblocks.diagnostics.verify import SimSpec
    from spinner_vs_spinner import critical_dt
    return SimSpec(
        name="Rigid blade tip load", run=tip_spec_run, params=dict(params), dt=2.5e-7,
        metrics=["tip_J_lat", "tip_F_lat_pk", "tip_Fz_pk"] + (["tip_J_z"] if z_case else ["tip_J_x"]),
        uncertain={k: tuple(v) for k, v in unc.items()},
        knockouts={"rigid H blade": (dict(flexH=0.0), {}), "no blade plasticity": (dict(plastic=0.0), {}),
                   "no blade friction": (dict(mu_bb=0.0), {}), "no gyroscopic coupling": (dict(gyro_on=0.0), {})},
        limits=[("robots never meet (1 ms run) -> no tip load", dict(t_end=0.001),
                 {"tip_J_lat": lambda v: abs(v) < 0.01})],      # noise floor of the arm settling under gravity
        bounds={"pen_ratio_max": (0.0, 0.5)}, critical_dt=critical_dt, mc_n=10, ledger_tol=0.05)


def main():
    T.apply()
    scale, fy0 = strength_scale()
    mc = pickle.load(open(os.path.join(HERE, "out", "spinner_vs_spinner", "mc.pkl"), "rb"))
    samples = mc["samples"]
    base = dict(H_sy_scale=scale, hist_dt=HIST_DT)
    cache = os.path.join(OUT, "runs.pkl")
    if os.path.exists(cache) and pickle.load(open(cache, "rb"))["scale"] == scale and "--rerun" not in sys.argv:
        R = pickle.load(open(cache, "rb"))["R"]
    else:
        from concurrent.futures import ProcessPoolExecutor
        t_ = time.time()
        run(dict(t_end=2e-4))
        with ProcessPoolExecutor() as ex:
            R = list(ex.map(_one, [(i, {**s, **base}) for i, s in enumerate(samples)], chunksize=2))
        pickle.dump(dict(scale=scale, R=R), open(cache, "wb"))
        print(f"{len(R)} engagements in {time.time()-t_:.0f}s")
    tu, Fu = half_sine_train(**USER, n_per=40)
    mu = pulse_metrics(tu, Fu)
    # worst VERIFIED engagement: the top candidates are re-run at dt/2 and dt/4; the exported load must hold to 2%
    def converged(r, z):
        # same criterion as verify_all's dt_convergence: Richardson-extrapolated error of the dt result < 1%
        lv = [tip_spec_run(r["params"], d) for d in (2.5e-7, 1.25e-7, 6.25e-8)]
        keys = ["tip_J_lat", "tip_F_lat_pk", "tip_Fz_pk"] + (["tip_J_z"] if z else ["tip_J_x"])
        for k in keys:
            v = [x[k] for x in lv]
            e1, e2 = abs(v[0] - v[1]), abs(v[1] - v[2])
            sc = max(abs(v[2]), 1e-12)
            if e1 <= 1e-12 * sc:
                continue
            p = math.log2(e1 / e2) if e2 > 0 else float("inf")
            vs = v[2] + (v[2] - v[1]) / (2 ** p - 1) if (math.isfinite(p) and p > 0.2) else v[2]
            if abs(v[0] - vs) / max(abs(vs), 1e-12) > 0.01:
                return False
        return True
    picks, skipped = {}, {}
    for name, key, z in (("overall", lambda r: r["all"]["J_lat"], False),
                         ("out-of-plane (Z)", lambda r: abs(r["z"]["J_z"]), True)):
        ranked = sorted(R, key=key, reverse=True)[:8]
        skipped[name] = []
        for r in ranked:
            if converged(r, z):
                picks[name] = r
                break
            skipped[name].append(r)
        else:
            picks[name] = ranked[0]
        print(name, "skipped (did not converge):", [(x["i"], round(key(x), 2)) for x in skipped[name]])
    print(f"strength scale {scale:.3f} (flap collapse {fy0:.0f} -> {F_BEND:.0f} N); user load {mu['impulse']:.3f} N s")
    exports, reps, cases = {}, {}, {}
    for name, r in picks.items():
        c = r["all"] if name == "overall" else r["z"]
        cases[name] = (r, c)
        print(f"worst {name}: sample {r['i']} arm {c['arm']}  J = ({c['J_x']:.3f}, {c['J_y']:.3f}, {c['J_z']:.3f}) N s, "
              f"bending |J| {c['J_lat']:.3f}, peak {c['F_pk']:.0f} N (Y {c['Fy_pk']:.0f}, Z {c['Fz_pk']:.0f}), pulses {c['n_pulses']}")
        rep = verify_all(tip_spec(r["params"], mc["unc"], z_case=(name != "overall")))
        print(rep.summary())
        for ck in rep.checks:
            if ck.status not in ("PASS",):
                print("   ", ck.status, ck.name, ck.detail[:400])
        reps[name] = rep
        # load window, then straight to zero within one sample, then hold zero to the end of the user's run
        tw = c["t"] - c["t"][0]
        tt = np.r_[tw, tw[-1] + HIST_DT, USER["t_end"]]
        cols = np.r_[np.c_[c["R"], c["L"], c["Z"]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
        cols[0] = 0.0
        bend = cols.copy(); bend[:, 0] = 0.0          # main table: bending components only
        t_tab, F_tab, info = fe_load_table(tt, bend, tol_frac=0.005)
        slug = "overall" if name == "overall" else "out_of_plane"
        notes = [f"Worst {name} tip load on a rigid Flapjack blade (simblocks spinner_vs_spinner, sample {r['i']}).",
                 "Arm frame at the tooth: X radial out, Y in-plane along the direction of rotation, Z out-of-plane up.",
                 f"Blade bends (full collapse) out-of-plane at {F_BEND:.0f} N tip load. Impulse Y {c['J_y']:+.3f} N s, Z {c['J_z']:+.3f} N s."]
        p1 = write_load_table(os.path.join(OUT, f"worst_{slug}_tip_load_table.txt"), t_tab, *F_tab.T,
                              notes=notes + ["Bending only: X set to 0 (radial is in the _dense_with_radial file)."])
        p2 = write_load_table(os.path.join(OUT, f"worst_{slug}_tip_load_dense_with_radial.txt"), tt, *cols.T,
                              notes=notes + [f"X radial impulse {c['J_x']:+.3f} N s is reliable; its PEAK is not (set by the "
                                             "contact stiffness, no tooth crushing modelled)."])
        # ---- Ansys tables in the user's frame: X side to side (+ right), Y forward/back (+ forward), Z up.
        # Blade pointing forward (+Y) at contact, spinning counterclockwise seen from above, so the in-plane
        # (along-rotation) component acts along -X. Radial (along the blade, Y) left at 0: its peak is not reliable.
        U = np.c_[-bend[:, 1], 0.0 * bend[:, 0], bend[:, 2]]
        tu_, Fu_, iu = fe_load_table(tt, U, tol_frac=0.02, max_points=40)
        Mu = np.c_[R_TIP * Fu_[:, 2], np.zeros(len(tu_)), -R_TIP * Fu_[:, 0]]   # M = r x F, r = (0, R_TIP, 0)
        hdr = [f"Worst {name} hit on Flapjack's blade (sim sample {r['i']}). Blade bends at {F_BEND:.0f} N tip load "
               "out-of-plane (AR600 full plastic + impact rate, worst case).",
               "Frame: X side to side (+ right), Y forward/back (+ forward), Z up. Blade pointing forward, spinning "
               "counterclockwise seen from above (flip X for the other direction).",
               f"Impulse: Fx {float(np.sum(0.5*(Fu_[1:,0]+Fu_[:-1,0])*np.diff(tu_))):+.3f} N s, Fz "
               f"{float(np.sum(0.5*(Fu_[1:,2]+Fu_[:-1,2])*np.diff(tu_))):+.3f} N s. Radial (Y) impulse {c['J_x']:+.2f} N s "
               "not included (peak unreliable)."]
        a1 = write_load_table(os.path.join(OUT, f"ANSYS_{slug}_force_at_tip.txt"), tu_, *Fu_.T,
                              notes=hdr + [f"Apply at the blade tip, {R_TIP*1e3:.0f} mm forward of the blade centre."])
        a2 = write_load_table(os.path.join(OUT, f"ANSYS_{slug}_force_at_centre.txt"), tu_, *Fu_.T,
                              notes=hdr + ["Apply at the blade centre together with the moment table (same rows)."])
        with open(os.path.join(OUT, f"ANSYS_{slug}_moment_at_centre.txt"), "w") as fh:
            for n_ in hdr + [f"Moment of the tip force about the blade centre (lever {R_TIP*1e3:.0f} mm), N m."]:
                fh.write(f"% {n_}\n")
            fh.write("% time_s\tMx_Nm\tMy_Nm\tMz_Nm\n")
            for row in zip(tu_, *Mu.T):
                fh.write("\t".join(f"{v:.6g}" for v in row) + "\n")
        exports[name] = dict(table=p1, dense=p2, points=info["points"], tt=tt, cols=cols, ansys=(a1, a2), urows=len(tu_))

    # ---------------- figures
    figs = []
    fig = plt.figure(figsize=(13, 7.4))
    top = T.header(fig, "Worst-case tip loads on a rigid blade",
                   f"Force the bending arm puts on the rest of Flapjack's blade, worst of {len(R)} engagements "
                   f"(blade bends out-of-plane at {F_BEND:.0f} N tip load, full collapse)", tag="SIMBLOCKS // LOAD EXPORT")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, bottom=0.14, top=top - 0.06, wspace=0.25, hspace=0.55,
                          width_ratios=[1.5, 1])
    for k, name in enumerate(picks):
        ax = fig.add_subplot(gs[k, 0]); e = exports[name]; tt, cols = e["tt"], e["cols"]
        if name != "overall":
            ax.plot(tu * 1e3, Fu, color=T.INK_3, lw=1.3, ls=(0, (4, 3)), label=f"current load, {mu['impulse']:.2f} N s")
        for ci, (lab, col) in enumerate([("X radial", 2), ("Y in-plane", 1), ("Z out-of-plane", 0)]):
            if name != "overall" and ci < 2:
                continue
            Jc = float(np.sum(0.5 * (cols[1:, ci] + cols[:-1, ci]) * np.diff(tt)))
            ax.plot(tt[:-1] * 1e3, cols[:-1, ci] / (1e3 if name == "overall" else 1), color=T.SERIES[col], lw=1.4,
                    label=f"{lab}, {Jc:+.2f} N s")
        ax.axhline(0, color=T.EDGE, lw=0.8)
        ax.set_xlim(0, max(1.0, min(5.0, tt[-2] * 1e3 * 1.3))); ax.set_xlabel("time after first blade load (ms)")
        ax.set_ylabel("tip force (kN)" if name == "overall" else "tip force (N)")
        ax.set_title(f"Worst {name}: sample {cases[name][0]['i']}")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
        ax.legend(loc="upper right", fontsize=7, ncol=3 if name == "overall" else 2); T.style_ticks(ax)
    ax2 = fig.add_subplot(gs[:, 1])
    J = np.array([r["all"]["J_lat"] for r in R]); P = np.array([r["all"]["F_pk"] for r in R])
    ax2.scatter(P / 1e3, J, s=16, color=T.SERIES[0], alpha=0.75, lw=0, label="engagements")
    for name, mk in (("overall", "D"), ("out-of-plane (Z)", "^")):
        r, c = cases[name]
        ax2.scatter([r["all"]["F_pk"] / 1e3], [r["all"]["J_lat"]], s=70, marker=mk, color=T.STATUS["FAIL"], lw=0,
                    label=f"worst {name}", zorder=4)
    ax2.scatter([F_BEND / 1e3], [mu["impulse"]], s=70, marker="s", color=T.INK_3, lw=0, label="current load", zorder=4)
    ax2.set_xlabel("peak tip load, in-plane + out-of-plane (kN)"); ax2.set_ylabel("bending impulse, in-plane + out-of-plane (N s)")
    ax2.set_title("All engagements"); ax2.legend(loc="lower right", fontsize=7); T.style_ticks(ax2)
    T.footer(fig, "Tip load = tooth spring force of the bending arm (elastic-plastic cantilever, first mode) on the rigid "
                  "blade, arm frame, while the blade is being hit (free ringing afterwards excluded). X: centrifugal pull removed, "
                  "averaged over the axial-mode period. Bending impulse = |(Jy, Jz)|. Tables keep each impulse exactly.")
    figs.append(T.save(fig, os.path.join(OUT, "tip_load")))

    b0 = build(dict(DEFAULTS))
    sch = Schematic("Where the rigid-blade tip load comes from",
                    "Flexible-blade sim -> force at the tip of the rigid blade in your MATLAB model")
    sch.body("V", "rotor", "Vertical blade", info=["bar, spinning", "flexible arms"])
    sch.body("tooth", "mass", "H tooth", info=[f"m = {b0['bots'][0]['mt']*1e3:.0f} g (1st mode)"])
    sch.body("hub", "rotor", "H rigid blade", info=["your MATLAB part", "load at the tip"])
    sch.body("frame", "mass", "Flapjack", info=["6-DOF + gyro"])
    sch.body("gnd", "ground", "")
    sch.link("V", "tooth", "contact", "Blade contact", "Hunt-Crossley\n+ friction", "penalty contact")
    sch.link("tooth", "hub", "elastoplastic", "Arm bending", f"Z yields at {F_BEND:.0f} N\nY, X stiffer", "beam_plastic_spring")
    sch.link("hub", "frame", "rigid", "Bearing", "spin DOF", "gyrostat_accel")
    sch.link("frame", "gnd", "spring", "Wheels/skid", "k = 3e5 N/m", "supports")
    sch.probe("hub", "EXPORTED: arm force on the rigid blade, at the tip")
    figs.append(sch.draw(os.path.join(OUT, "load_schematic")))

    r, c = cases["overall"]; rz, cz = cases["out-of-plane (Z)"]
    pk = dict(label="Worst bending impulse at tip", value=f"{c['J_lat']:.2f}", unit="N s", status="INFO",
              sub=f"peak {c['F_pk']/1e3:.1f} kN in-plane; worst out-of-plane {abs(cz['J_z']):.2f} N s, {cz['Fz_pk']:.0f} N; "
                  f"current load {mu['impulse']:.2f} N s")
    e = exports["overall"]
    hist = dict(t=e["tt"][:-2], X=e["cols"][:-2, 0], Y=e["cols"][:-2, 1], Z=e["cols"][:-2, 2])
    ez = exports["out-of-plane (Z)"]
    hist["Zz"] = np.interp(hist["t"], ez["tt"], ez["cols"][:, 2])
    panels = [dict(title="Worst overall: Y in-plane", keys=["Y"], unit="N", color=1, t_scale=1e3, t_label="time (ms)"),
              dict(title="Worst overall: X radial (centrifugal removed)", keys=["X"], unit="N", color=2, t_scale=1e3,
                   t_label="time (ms)"),
              dict(title="Worst overall: Z out-of-plane", keys=["Z"], unit="N", t_scale=1e3, t_label="time (ms)"),
              dict(title="Worst out-of-plane case: Z", keys=["Zz"], unit="N", t_scale=1e3, t_label="time (ms)")]
    findings = [
        f"Worst overall (sample {r['i']}): bending impulse {c['J_lat']:.2f} N s at the tip, signed components X "
        f"{c['J_x']:+.2f}, Y {c['J_y']:+.2f}, Z {c['J_z']:+.2f} N s; peak {c['F_pk']:.0f} N, almost all in-plane "
        f"(Y {c['Fy_pk']:.0f} N). The current load (two half-sines, Z only) carries {mu['impulse']:.2f} N s at "
        f"{F_BEND:.0f} N.",
        ("Unconverged engagements set aside (their load changes with time step): "
         + "; ".join(f"{nm}: " + ", ".join(f"sample {x['i']}" for x in sk) for nm, sk in skipped.items() if sk)
         + f". Largest of them: bending impulse {max(x['all']['J_lat'] for sk in skipped.values() for x in sk):.2f} N s."
         if any(skipped.values()) else
         "Every top-8 candidate in both rankings converged with time step (harness criterion, < 1%)."),
        f"Worst out-of-plane (sample {rz['i']}): Z impulse {cz['J_z']:+.2f} N s, Z peak {cz['Fz_pk']:.0f} N, "
        f"{cz['n_pulses']} pulse(s).",
        f"Why in-plane governs: the arm is {b0['bots'][0]['W']*1e3:.0f} mm wide and {b0['bots'][0]['t']*1e3:.2f} mm "
        f"thick, so it carries about (W/t)^2 = {(b0['bots'][0]['W']/b0['bots'][0]['t'])**2:.0f}x more stiffness "
        f"and {(b0['bots'][0]['W']/b0['bots'][0]['t']):.1f}x more yield load in-plane. A horizontal blade is hit "
        "mostly in its own plane.",
        f"Tables (bending only, X = 0): {exports['overall']['points']} and {exports['out-of-plane (Z)']['points']} points, "
        f"impulse kept exactly. Dense files at {HIST_DT*1e6:.1f} us include radial X: its impulse is reliable "
        f"({c['J_x']:+.2f} N s worst overall) but its peak is set by the contact stiffness (no tooth crushing).",
    ]
    assumptions = [
        "Ranking uses the bending components (in-plane Y + out-of-plane Z). Radial X is exported too: in jammed "
        "engagements the tooth pushes inward along the arm for several ms (up to about 13 N s), which loads the hub "
        "and mount axially but does not bend the blade.",
        f"Flapjack's blade strength scaled x{scale:.2f} so the most it can pass out-of-plane (full plastic collapse with "
        f"hardening) is {F_BEND:.0f} N at the tip (your bending load); in-plane strength scales with it. Blade 30 x 6.35 mm, "
        "190 mm tip to tip, arms bend from a 12.5 mm clamp radius (82.5 mm lever). AR500 curve shape, stiffness unchanged.",
        "Engagements reuse the study's inputs (phases, closing 2-4 m/s, offset, heading, restitution, friction, rpm); "
        "the vertical is the same placeholder bar, both blades bend.",
        "Load = the arm's first-mode spring force on the rigid blade. Higher bending modes, torsion and tooth "
        "crushing (which would cap the in-plane peak lower) are not modelled.",
        "Arm frame: X radial out, Y along the direction of rotation, Z up.",
    ]
    rep = reps["overall"]
    paths = results_report(OUT, "Rigid blade tip load", "What is the most aggressive tip load to apply to a rigid "
                           "Flapjack blade, from the flexible-blade sims?", pk, rep, history=hist, history_panels=panels,
                           findings=findings, assumptions=assumptions,
                           sources=["spinner_vs_spinner.py", "beam_plastic_spring", "fe_load_table"],
                           extra_svgs=[f[1] for f in figs])
    print(paths); [print("-", f) for f in findings]
    return paths, exports

if __name__ == "__main__":
    main()

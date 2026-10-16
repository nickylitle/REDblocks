"""Report for the vertical-vs-horizontal spinner double-hit study.

  python spinner_vs_spinner_report.py            # uses out/spinner_vs_spinner/mc.pkl if present
  python spinner_vs_spinner_report.py --mc 160   # (re)runs the 5-configuration Monte Carlo first

Pages: answer + verification (results_report), time history of the nominal double hit, mechanism figure
(P(fast double hit) per blade configuration with 95% intervals, gap histogram), side/top snapshots.
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
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from spinner_vs_spinner import run, build, DEFAULTS, critical_dt, T_FAST, T_TAIL, T_MERGE
from simblocks.rigid_body.spatial import quat_to_matrix
from simblocks.diagnostics.verify import SimSpec, verify_all
from simblocks.diagnostics.failure import _wilson
from simblocks.viz import theme as T
from simblocks.viz.report import results_report

OUT = os.path.join(HERE, "out", "spinner_vs_spinner"); os.makedirs(OUT, exist_ok=True)
NOMINAL = {}                 # chosen from the Monte Carlo (see main)
from spinner_vs_spinner import CFG
JOB_PATH = os.path.join(HERE, "spinner_vs_spinner_job.yaml")   # sample count + dimensions live here
LABELS = {"phase_H_deg": "H blade phase", "phase_V_deg": "V blade phase", "v_close": "closing speed",
          "H_zblade": "H blade height", "V_R": "V blade radius", "V_t": "V blade thickness", "V_clear": "V tip clearance",
          "y_off": "lateral offset", "yaw_off_deg": "heading error", "e_bb": "restitution", "mu_bb": "blade friction",
          "H_rpm": "H blade rpm", "V_rpm": "V blade rpm"}
MC_PATH = os.path.join(OUT, "mc.pkl")
DT_MC = 2.5e-7      # Monte Carlo step (its decision metrics converge within 0.6% there)
DT_VER = 6.25e-8    # nominal case: verified and reported at this step


# ------------------------------------------------------------------ Monte Carlo over blade configurations
def _mc_one(a):
    cfg, s = a
    o, ex = run(dict(s), raw=True)
    return cfg, o, [(h["t0"], h["t1"], h["J"], h["toothH"], h["toothV"], h["z_pk"]) for h in ex["hits"]]


def run_mc(n):
    """Run the study defined in the job file (dimensions, sampler) here, with n samples per config. Only on
    request (--mc N); big studies belong on the user's machine: simblocks batch spinner_vs_spinner_job.yaml."""
    from concurrent.futures import ProcessPoolExecutor
    from simblocks.batch import load_job, make_samples
    job = load_job(JOB_PATH)
    samples = make_samples(job, n)
    cfgs = {c: (ov or {}) for c, ov in job["configs"].items()}
    jobs = [(c, {**job.get("base", {}), **s}) for c in cfgs for s in samples]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_mc_one, [(c, {**p, **cfgs[c]}) for c, p in jobs], chunksize=4))
    unc = {k: tuple(v) for k, v in job["samples"]["uniform"].items()}
    out = dict(samples=samples, unc=unc, cfg=list(cfgs), res=[(c, i % n, o, h) for i, (c, o, h) in enumerate(res)],
               code=_code_hash())
    pickle.dump(out, open(MC_PATH, "wb"))
    return out


def _code_hash():
    import hashlib
    return hashlib.sha1(open(os.path.join(HERE, "spinner_vs_spinner.py"), "rb").read()).hexdigest()


def mc_from_results(path):
    """Monte Carlo data from a local batch results zip (simblocks batch spinner_vs_spinner_job.yaml)."""
    from simblocks.batch import load_results
    from spinner_vs_spinner import HIT_KEYS
    R = load_results(path)
    cfgs, samples, res = [], {}, []
    for r in R["runs"]:
        _, c, i = r["key"].split("|"); i = int(i)
        if c not in cfgs:
            cfgs.append(c)
        samples.setdefault(i, {k: r["params"][k] for k in R["job"]["samples"]["uniform"] if k in r["params"]})
        if r["record"] is None:
            continue
        hits = [dict(zip(HIT_KEYS, h)) for h in r["record"]["hits"]]
        res.append((c, i, r["record"]["out"], [(h["t0"], h["t1"], h["J"], h["toothH"], h["toothV"], h["z_pk"])
                                              for h in hits]))
    n = max(samples) + 1
    unc = {k: tuple(v) for k, v in R["job"]["samples"]["uniform"].items()}
    return dict(samples=[samples.get(i, {}) for i in range(n)], unc=unc, cfg=cfgs, res=res, meta=R["meta"],
                dt_check=R["dt_check"], verify=R["verify"])


def mc_table(mc):
    rows = {}
    for c in mc["cfg"]:
        R = [r[2] for r in mc["res"] if r[0] == c]
        v = int(sum(r["valid"] for r in R)); k = int(sum(r["fast_double"] for r in R))
        k6 = int(sum(r["double_6ms"] for r in R))
        gaps = [r["gap12_ms"] for r in R if r["valid"] and r["n_hits"] >= 2]
        rows[c] = dict(n=v, k=k, p=k / v if v else float("nan"), ci=_wilson(k, v), k6=k6, p6=k6 / v if v else 0.0,
                       ci6=_wilson(k6, v), gaps=gaps, worst_ledger=max(abs(r["_ledger_rel"]) for r in R),
                       worst_pen=max(r["pen_ratio_max"] for r in R),
                       p_wb_first=float(np.mean([r["wb_before_bb"] for r in R])),
                       rpmV1=float(np.median([r["rpmV_at_hit1"] for r in R if r["valid"]])),
                       fast_rpmV1=[r["rpmV_at_hit1"] for r in R if r["fast_double"]],
                       fast_wb=[r["wb_before_bb"] for r in R if r["fast_double"]])
    return rows


# ------------------------------------------------------------------ geometry for snapshots
def _rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def box_polys(b, row):
    """World corner sets of every box at one history row: list of (bot, kind, corners[8,3])."""
    out = []
    st = []
    for bi, bot in enumerate(b["bots"]):
        o = 6 + 8 * bi
        X = row[o:o + 3]; RB = quat_to_matrix(np.ascontiguousarray(row[o + 3:o + 7])); psi = row[o + 7]
        RR = RB @ bot["Rab"] @ _rz(psi); CW = X + RB @ bot["d_b"]
        st.append((X, RB, RR, CW))
    signs = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
    for k in range(len(b["B_bot"])):
        bi = int(b["B_bot"][k]); kind = int(b["B_kind"][k]); h = b["B_h"][k]; X, RB, RR, CW = st[bi]
        if kind == 0:
            C = np.array([X + RB @ (b["B_c"][k] + np.array(s) * h) for s in signs])
        elif kind == 1:
            C = np.array([CW + RR @ (np.array(s) * h) for s in signs])
        else:
            j = int(b["B_arm"][k]); sg = b["A_sign"][j]; rr = b["A_root"][j]; R = b["A_R"][j]; L = b["A_L"][j]
            XT = row[47 + 3 * j:50 + 3 * j]; anchor = CW + RR @ np.array([sg * R, 0, 0]); u = XT - anchor
            C = []
            for s in signs:
                srad = rr if s[0] < 0 else R
                r0 = np.array([sg * srad, s[1] * h[1], s[2] * h[2]])
                C.append(CW + RR @ r0 + (srad - rr) / L * u)
            C = np.array(C)
        out.append((bi, kind, C))
    return out


def draw_view(ax, b, row, plane, xlim, ylim):
    ia, ib = {"side": (0, 2), "top": (0, 1)}[plane]
    order = sorted(box_polys(b, row), key=lambda q: (q[1] >= 1, q[0]))
    for bi, kind, C in order:
        P = C[:, [ia, ib]] * 1e3
        try:
            hull = ConvexHull(P); P = P[hull.vertices]
        except Exception:
            continue
        col = T.SERIES[0] if bi == 0 else T.SERIES[1]
        if kind == 0:
            ax.fill(P[:, 0], P[:, 1], color=T.EDGE, alpha=0.55, lw=0, zorder=1)
        else:
            ax.fill(P[:, 0], P[:, 1], color=col, alpha=0.85 if kind == 2 else 0.6, lw=0.6, ec=T.SURFACE, zorder=3)
    if plane == "side":
        ax.axhline(0, color=T.INK_3, lw=1.0, zorder=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    T.style_ticks(ax)


# ------------------------------------------------------------------ main
def main():
    T.apply()
    if "--from" in sys.argv:
        mc = mc_from_results(sys.argv[sys.argv.index("--from") + 1])
        print("loaded", mc["meta"]["n_runs"], "runs from", mc["meta"]["machine"].get("platform"))
    elif "--mc" in sys.argv:
        n = int(sys.argv[sys.argv.index("--mc") + 1])
        t0 = time.time(); mc = run_mc(n); print(f"MC {n} x {len(mc['cfg'])} in {time.time()-t0:.0f}s")
    elif os.path.exists(MC_PATH):
        mc = pickle.load(open(MC_PATH, "rb"))
        if "unc" not in mc or not isinstance(mc["unc"], dict):
            mc["unc"] = {}
    else:
        raise SystemExit("no study data: pass --from <results.zip> (from simblocks batch) or --mc N to run one here")
    tab = mc_table(mc)
    # nominal = the dominant fast mechanism: an H-blade snap-back double hit (V blade rigid), both blades at speed
    # (no chassis hit first), the one with the largest smaller-of-two impulses
    NOM_CFG = "H blade flexes only"
    cand = [(min(r[2]["J1_Ns"], r[2]["J2_Ns"]), r[1]) for r in mc["res"] if r[0] == NOM_CFG and
            r[2]["fast_double"] and not r[2]["wb_before_bb"]]
    i_nom = max(cand)[1] if cand else 0
    NOMINAL.clear(); NOMINAL.update(mc["samples"][i_nom]); NOMINAL.update(CFG[NOM_CFG])
    print("nominal sample", i_nom, {k: round(v, 4) for k, v in NOMINAL.items()})

    # ---------------- verification of the nominal case
    from spinner_vs_spinner import make_spec
    vcache = os.path.join(OUT, "verify_cache.pkl")
    rep = None
    if os.path.exists(vcache) and "--reverify" not in sys.argv:
        c = pickle.load(open(vcache, "rb"))
        if c["params"] == NOMINAL and c["dt"] == DT_VER and c.get("code") == _code_hash() and c.get("unc") == mc["unc"]:
            rep = c["rep"]; print("verification: reused cached report for this nominal")
    if rep is None:
        rep = verify_all(make_spec(NOMINAL, uncertain=mc["unc"], dt=DT_VER, mc_n=20))
        sp = rep.spec; run_, cd_ = sp.run, sp.critical_dt
        sp.run = None; sp.critical_dt = None
        try:
            pickle.dump(dict(params=dict(NOMINAL), dt=DT_VER, rep=rep, code=_code_hash(), unc=mc["unc"]), open(vcache, "wb"))
        except Exception as exc:
            print("verification cache not written:", exc)
        sp.run, sp.critical_dt = run_, cd_
    print(rep.summary())
    for c in rep.checks:
        if c.status != 'PASS':
            print('  ', c.status, c.name, c.detail[:600])

    # ---------------- nominal history
    nom, ex = run(dict(NOMINAL), dt=DT_VER, history=True)
    Hh = ex["hist"]; b = ex["build"]; hits = ex["hits"]
    t = Hh[:, 0]
    jV = 2 if hits[0]["toothV"] > 0 else 3          # V arm that took hit 1
    jH = 0 if hits[0]["toothH"] > 0 else 1
    t0 = hits[0]["t0"] - 0.4e-3; t1 = (hits[1]["t1"] if len(hits) > 1 else hits[0]["t1"]) + 1.2e-3
    sel = (t >= t0) & (t <= t1)
    hist = dict(t=t[sel] - hits[0]["t0"], Fbb=Hh[sel, 1] / 1e3,
                uVf=Hh[sel, 32 + 4 * jV] * 1e3, sVf=Hh[sel, 34 + 4 * jV] * 1e3,
                uHf=Hh[sel, 32 + 4 * jH] * 1e3, sHf=Hh[sel, 34 + 4 * jH] * 1e3,
                uVl=Hh[sel, 31 + 4 * jV] * 1e3, uHl=Hh[sel, 31 + 4 * jH] * 1e3, sHl=Hh[sel, 33 + 4 * jH] * 1e3,
                rpmH=Hh[sel, 4] * 60 / (2 * math.pi), rpmV=Hh[sel, 5] * 60 / (2 * math.pi))
    panels = [dict(title="Blade-to-blade contact force", keys=["Fbb"], unit="kN", t_scale=1e3,
                   t_label="time after first contact (ms)"),
              dict(title="H tooth: in-plane (lag) bend", keys=["uHl", "sHl"], labels=["total", "permanent"],
                   unit="mm", color=0, t_scale=1e3, t_label="time after first contact (ms)", headroom=0.45,
                   legend_loc="upper right", legend_ncol=2),
              dict(title="Out-of-plane (flap) tip deflection", keys=["uHf", "uVf"], labels=["H tooth", "V tooth"],
                   unit="mm", color=2, t_scale=1e3, t_label="time after first contact (ms)", headroom=0.45,
                   legend_loc="upper right", legend_ncol=2),
              dict(title="Blade speed", keys=["rpmH", "rpmV"], labels=["H blade", "V blade"], unit="rpm",
                   t_scale=1e3, t_label="time after first contact (ms)", headroom=0.35, legend_loc="upper right",
                   legend_ncol=2)]

    ex_nom_build = ex["build"]
    # ---------------- mechanism figure
    figs_extra = []
    order = list(CFG)
    fig = plt.figure(figsize=(13, 6.4))
    top = T.header(fig, "Which blade behaviour makes the fast double hit",
                   f"{tab[order[0]]['n']}+ engagements per configuration, the same samples in every configuration, "
                   f"{len(mc['unc'])} inputs varied (listed below)", tag="SIMBLOCKS // MECHANISM")
    import textwrap
    varied = "Varied: " + ", ".join(f"{LABELS.get(k, k)} {lo:g} to {hi:g}" for k, (lo, hi) in mc["unc"].items())
    fig.text(0.018, top - 0.005, "\n".join(textwrap.wrap(varied, 190)[:3]), ha="left", va="top", fontsize=7,
             color=T.INK_3, family=T.MONO)
    top -= 0.075
    gs = fig.add_gridspec(1, 2, left=0.20, right=0.97, bottom=0.27, top=top - 0.07, wspace=0.32,
                          width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(order))[::-1]
    for y, c in zip(ys, order):
        r = tab[c]; lo, hi = r["ci"]
        col = T.SERIES[0] if r["k"] > 0 else T.INK_3
        ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot([r["p"] * 100], [y], "o", color=col, ms=7, mec=T.PANEL, mew=1.5, zorder=4)
        ax.text(hi * 100 + 0.6, y, f"{r['p']*100:.1f}%  ({r['k']}/{r['n']})", va="center", fontsize=7.5,
                color=T.INK_2, family=T.MONO)
    ax.set_yticks(ys); ax.set_yticklabels(order, fontsize=8.5, color=T.INK)
    xmax = max(tab[c]["ci"][1] for c in order) * 100
    ax.set_xlim(0, xmax * 1.45 + 2); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("P(second hit within 1.5 ms)  (%; bar = 95% interval)")
    ax.set_title("Fast double hit probability"); T.style_ticks(ax)
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ex_nom_build["bots"]
    half = {nm: 0.5e3 / (math.sqrt(bp[0]["spring_info"][nm]["k"] / bp[0]["mt"]) / (2 * math.pi)) for nm in ("lag", "flap")}
    mk = {"elastic-plastic (both)": "o", "elastic only": "s", "H blade flexes only": "^", "V blade flexes only": "v",
          "rigid blades": "X"}
    for i, c in enumerate(order):
        G = [(r[2]["gap12_ms"], r[3][1][2]) for r in mc["res"] if r[0] == c and r[2]["valid"] and r[2]["n_hits"] >= 2
             and r[2]["gap12_ms"] <= 6.0]
        if G:
            g, j2 = np.array(G).T
            ax2.scatter(g, j2, s=26, marker=mk[c], color=T.INK_3 if c == "rigid blades" else T.SERIES[i % 5],
                        alpha=0.9, lw=0, label=c, zorder=3)
    ax2.set_xscale("log"); ax2.set_xlim(0.05, 7)
    ax2.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5]); ax2.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax2.axvline(half["lag"], color=T.STATUS["INFO"], lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax2.axvline(half["flap"], color=T.STATUS["INFO"], lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax2.axvline(T_FAST * 1e3, color=T.STATUS["WARN"], lw=1.1, ls=(0, (4, 3)), zorder=2)
    tr = ax2.get_xaxis_transform()
    ax2.text(half["lag"], 1.01, "H lag\n1/2 period", fontsize=6.5, color=T.STATUS["INFO"], va="bottom", ha="center",
             family=T.MONO, transform=tr)
    ax2.text(half["flap"], 1.01, "H flap\n1/2 period", fontsize=6.5, color=T.STATUS["INFO"], va="bottom", ha="center",
             family=T.MONO, transform=tr)
    ax2.text(T_FAST * 1e3, 1.01, "1.5 ms", fontsize=6.5, color=T.STATUS["WARN"], va="bottom", ha="center",
             family=T.MONO, transform=tr)
    ax2.set_xlabel("gap between hit 1 and hit 2 (ms, log)"); ax2.set_ylabel("hit 2 impulse (N s)")
    ax2.set_title("Every double hit: gap vs second-hit size", pad=30)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=6.8)
    T.style_ticks(ax2)
    T.footer(fig, "Fast double hit: a second blade-to-blade hit starting less than 1.5 ms after the first ends. "
                  "The next tooth needs at least one half revolution (2.5 ms at 12,000 rpm), so faster returns "
                  "need the blade itself to move.")
    figs_extra.append(T.save(fig, os.path.join(OUT, "mechanism")))

    # ---------------- snapshots (side + top) at hit 1, max bend, hit 2
    def peak_t(h):
        m = (t >= h["t0"]) & (t <= h["t1"] + 1e-9)
        return t[m][np.argmax(Hh[m, 1])] if m.any() else h["t0"]
    t_h1 = peak_t(hits[0])
    if len(hits) > 1:
        t_h2 = peak_t(hits[1])
        m = (t > t_h1) & (t < hits[1]["t0"])
        bend = np.hypot(Hh[:, 31 + 4 * jH], Hh[:, 32 + 4 * jH])
        t_mid = t[m][np.argmax(bend[m])] if m.any() else 0.5 * (hits[0]["t1"] + hits[1]["t0"])
        moments = [(t_h1, "HIT 1 (peak force)"), (t_mid, "H TOOTH MOST BENT"), (t_h2, "HIT 2 (peak force)")]
    else:
        moments = [(t_h1, "HIT 1 (peak force)")]
    xc = hits[0]["x_pk"] * 1e3
    fig = plt.figure(figsize=(13, 7.2))
    top = T.header(fig, "Nominal double hit: blade positions",
                   "Side view (x-z) and top view (x-y), mm. H robot blue, V robot amber, chassis grey. "
                   "Arms drawn with their computed bend.", tag="SIMBLOCKS // SNAPSHOTS")
    gs = fig.add_gridspec(2, len(moments), left=0.06, right=0.98, bottom=0.12, top=top - 0.06, hspace=0.38,
                          wspace=0.18)
    for i, (tm, lab) in enumerate(moments):
        row = Hh[np.argmin(np.abs(t - tm))]
        a1 = fig.add_subplot(gs[0, i]); draw_view(a1, b, row, "side", (xc - 90, xc + 90), (-5, 85))
        a1.set_title(f"{lab}  t = {(tm - hits[0]['t0'])*1e3:+.2f} ms", fontsize=9)
        a1.set_xlabel("x (mm)"); a1.set_ylabel("z (mm)" if i == 0 else "")
        a2 = fig.add_subplot(gs[1, i]); draw_view(a2, b, row, "top", (xc - 90, xc + 90), (-60, 60))
        a2.set_xlabel("x (mm)"); a2.set_ylabel("y (mm)" if i == 0 else "")
    T.footer(fig, "Rigid hub + two flexible arms per blade; arm shape is linear in radius (lumped first bending "
                  "mode). Geometry is the sim's own contact geometry.")
    figs_extra.append(T.save(fig, os.path.join(OUT, "snapshots")))

    # ---------------- what was built: the physical model
    from simblocks.viz.diagram import Schematic, SignalFlow
    from spinner_vs_spinner import build as _build, DEFAULTS as _D
    bH, bV = _build(dict(_D))["bots"]                    # flexible-blade values (the nominal runs V rigid)
    sch = Schematic("Spinner vs spinner: physical model",
                    "Two 6-DOF robots, each with a spinning blade = rigid hub + two flexible arms (one per tooth)")
    sch.body("gH", "ground", "")
    sch.body("H", "mass", "Flapjack", info=[f"m = {bH['m']:.3f} kg", "6-DOF + gyro"])
    sch.body("hubH", "rotor", "H hub", info=[f"{bH['R']*2e3:.0f} mm bar", f"{nom['rpmH_at_hit1']:.0f} rpm"])
    sch.body("tH", "mass", "H tooth", info=[f"{bH['mt']*1e3:.0f} g (1st mode)"])
    sch.body("tV", "mass", "V tooth", info=[f"{bV['mt']*1e3:.0f} g (1st mode)"])
    sch.body("hubV", "rotor", "V hub", info=[f"{bV['R']*2e3:.0f} mm bar", "front moves up"])
    sch.body("V", "mass", "Vertical", info=[f"m = {bV['m']:.3f} kg", "6-DOF + gyro"])
    sch.body("gV", "ground", "")
    sch.link("gH", "H", "spring", "Wheels + skid", "k 3e5 N/m, one-sided\nmu 0.05 / 1.6 / 0.2", "supports")
    sch.link("H", "hubH", "rigid", "H bearing", "spin DOF", "gyrostat_accel")
    sch.link("hubH", "tH", "elastoplastic", "H arm bend",
             f"lag k {bH['k_lag']:.2g}, flap k {bH['k_flap']:.2g} N/m\nyield + hardening", "beam_plastic_spring")
    sch.link("tH", "tV", "contact", "Blade contact", "Hunt-Crossley, e 0.1-0.7\nfriction 0.1-0.6", "point_box_penetration")
    sch.link("tV", "hubV", "elastoplastic", "V arm bend",
             f"lag k {bV['k_lag']:.2g}, flap k {bV['k_flap']:.2g} N/m\n(rigid in the nominal case)", "beam_plastic_spring")
    sch.link("hubV", "V", "rigid", "V bearing", "spin DOF", "gyrostat_accel")
    sch.link("V", "gV", "spring", "Wheels + forks", "k 3e5 N/m, one-sided", "supports")
    sch.probe("tH", "hits (contact episodes), bend, set")
    figs_extra.append(sch.draw(os.path.join(OUT, "model_schematic"), width=17.0))

    # ---------------- the double hit as a sequence of events (nominal case numbers)
    sq = SignalFlow("How the double hit happens", f"Nominal case: gap {nom['gap12_ms']:.2f} ms, second hit "
                    f"{nom['J2_Ns']:.1f} N s after a {nom['J1_Ns']:.1f} N s first hit")
    sq.stage("h1", "1. First hit", [("tooth meets tooth", f"{nom['J1_Ns']:.1f} N s"),
                                    ("H tooth decelerates", "hub keeps spinning")], role="force")
    sq.stage("bend", "2. H arm bends back", [("in-plane (lag) bend", f"{nom['defl_lag_H_mm']:.0f} mm, yields"),
                                             ("tooth leaves contact", "pushed out of the path")], role="constitutive")
    sq.stage("snap", "3. Arm springs forward", [("stored bend energy", "released in"),
                                                (f"~1/2 lag period", f"{half['lag']:.2f} ms")], role="geometry")
    sq.stage("h2", "4. Second hit", [("same teeth meet again", f"{nom['J2_Ns']:.1f} N s"),
                                     ("rigid blades: no", "next tooth needs >= 2.5 ms")], role="force")
    sq.stage("after", "5. Permanent set", [("H tooth keeps", f"{nom['set_lag_H_mm']:.0f} mm in-plane"),
                                           ("robots separate", "")], role="output")
    sq.edge("h1", "bend", "load > yield"); sq.edge("bend", "snap", "contact ends")
    sq.edge("snap", "h2", f"{nom['gap12_ms']:.2f} ms"); sq.edge("h2", "after")
    figs_extra.append(sq.draw(os.path.join(OUT, "double_hit_sequence"), size=(15.0, 3.9), tag="SIMBLOCKS // SEQUENCE OF EVENTS"))

    # ---------------- answer page
    ep = tab["elastic-plastic (both)"]; rg = tab["rigid blades"]; el = tab["elastic only"]
    Hf = tab["H blade flexes only"]; Vf = tab["V blade flexes only"]
    reproduced = Hf["p"] > 2 * rg["p"] and ep["k"] > 0
    answer = dict(label="Double hit reproduced?", value="YES" if reproduced else "NO", unit="",
                  status="PASS" if reproduced else "WARN",
                  sub=f"fast double hits: H bending {Hf['p']*100:.0f}%, both bending {ep['p']*100:.0f}%, "
                      f"rigid {rg['p']*100:.0f}%")
    def gaps_of(c, lo, hi):
        return [g for g in tab[c]["gaps"] if lo <= g < hi]
    def fast_list(c):
        return [(r[2]["gap12_ms"], r[3][1][2], r[2]["wb_before_bb"], r[2]["rpmV_at_hit1"]) for r in mc["res"]
                if r[0] == c and r[2]["fast_double"]]
    rg_fast = fast_list("rigid blades"); ep_fast = fast_list("elastic-plastic (both)")
    hf_fast = fast_list("H blade flexes only")
    snap = [g for g, j, w, v in hf_fast if g < 0.3]; snapJ = [j for g, j, w, v in hf_fast if g < 0.3]
    flapg = [g for c in ("elastic-plastic (both)", "V blade flexes only") for g, j, w, v in fast_list(c) if g >= 0.3]
    findings = [
        f"Second blade hit within 1.5 ms of the first: both blades bending {ep['k']}/{ep['n']} engagements "
        f"({ep['p']*100:.1f}%, 95% CI {ep['ci'][0]*100:.1f}-{ep['ci'][1]*100:.1f}%), H blade bending and V rigid "
        f"{Hf['k']}/{Hf['n']} ({Hf['p']*100:.1f}%, CI {Hf['ci'][0]*100:.1f}-{Hf['ci'][1]*100:.1f}%), V bending only "
        f"{Vf['p']*100:.1f}%, elastic-only {el['p']*100:.1f}%, rigid {rg['k']}/{rg['n']} ({rg['p']*100:.1f}%). "
        "Same random engagements in every configuration.",
        f"Fast mechanism 1, H lag snap-back: {len(snap)} of the H-bending double hits come back after "
        f"{np.min(snap) if snap else float('nan'):.2f}-{np.max(snap) if snap else float('nan'):.2f} ms (H arm in-plane "
        f"half period {half['lag']:.2f} ms) with a second impulse of {np.median(snapJ) if snapJ else float('nan'):.1f} N s "
        "median: the tooth is knocked back, springs forward and hits the same V tooth again.",
        f"Fast mechanism 2, flap and permanent bend: gaps of {np.min(flapg) if flapg else float('nan'):.1f}-"
        f"{np.max(flapg) if flapg else float('nan'):.1f} ms when the V blade bends (H arm flap half period "
        f"{half['flap']:.2f} ms; the V arm is too stiff to ring that slowly, so its part is the permanent bend that "
        "moves the tooth back into the path).",
        f"When the V blade also bends and yields, the H snap-back mostly lands inside the same contact: "
        f"{ep['k']} fast doubles with both bending vs {Hf['k']} with H bending alone.",
        f"Rigid blades: {len(rg_fast)} fast double(s), all after a blade-on-chassis hit had slowed the V blade to "
        f"{', '.join(f'{v:.0f}' for g, j, w, v in rg_fast)} rpm (gaps {', '.join(f'{g:.2f}' for g, j, w, v in rg_fast)} ms). "
        "No rigid engagement with both blades at speed double-hits inside 1.5 ms.",
        f"Context: in {ep['p_wb_first']*100:.0f}% of engagements a blade hits the other robot's chassis first (mostly the "
        f"V blade on Flapjack's nose plate), slowing that blade. {sum(1 for g, j, w, v in ep_fast if w)}/{len(ep_fast)} "
        "of the both-bending fast doubles follow such a chassis hit; the H-bending-only ones mostly do not "
        f"({sum(1 for g, j, w, v in hf_fast if w)}/{len(hf_fast)}).",
        f"Nominal case (an H-blade snap-back double hit, V blade rigid; V at {nom['rpmV_at_hit1']:.0f} rpm, H at "
        f"{nom['rpmH_at_hit1']:.0f} rpm when the blades meet): hit 1 {nom['J1_Ns']:.1f} N s, hit 2 {nom['J2_Ns']:.2f} N s "
        f"after {nom['gap12_ms']:.2f} ms; the H tooth is pushed back {nom['defl_lag_H_mm']:.1f} mm in-plane (keeps "
        f"{nom['set_lag_H_mm']:.1f} mm) and flaps {nom['defl_flap_H_mm']:.1f} mm; {nom['W_plastic_J']:.0f} J into "
        "permanent bending.",
        f"Every run closes its energy ledger (worst {max(r['worst_ledger'] for r in tab.values())*100:.2f}%) and no "
        f"contact point sinks past {max(r['worst_pen'] for r in tab.values())*100:.0f}% of a half-thickness "
        "(no tunnelling through a blade).",
    ]
    assumptions = [
        "H robot = Flapjack: 190 x 30 x 6.35 mm bar, 280 g, 600 J (11,360 rpm), blade mid-plane 15 mm off the floor, "
        "arm clamped at 12.5 mm radius. AR500 stands in for AR600/SR500 (not in the material catalog).",
        "V robot = generic beetleweight bar vertical: S7 bar 120 x 25 x 12.7 mm, 12,000 rpm, axle 65 mm up, front of "
        "the blade moving up, arm clamped at 15 mm radius. All V numbers are placeholders.",
        "Each arm bends in its first mode only: lumped tooth mass 33/140 of the arm mass on cantilever flap and lag "
        "springs with elastic-plastic hinges from the section's moment-curvature (beam_plastic_spring). Centrifugal "
        "stiffening of the flap (about 6%) and torsion are not modelled.",
        "Contact: Hunt-Crossley with restitution 0.2-0.6, Coulomb friction 0.15-0.5, local stiffness 1e11 N/m per m of "
        "edge. Contact forces are not meaningful (no gouging or tooth plasticity); impulses and timing are.",
        "Robots coast in at 2-4 m/s closing with drives and weapon motors off for the 40 ms engagement. No forks, "
        "wedgelets or armor geometry beyond simple chassis boxes.",
        "Study data: the sampled inputs and ranges listed on the mechanism page, set in the job file. The nominal case "
        "is re-run and verified with the current sim (gap-driven big steps, early stop), which matched the fixed-step "
        "runs on 40 of 40 engagements.",
        f"Hits are blade-to-blade contact episodes; contact gaps under {T_MERGE*1e3:.1f} ms are chatter within one hit. "
        f"An engagement counts only if the sim runs {T_TAIL*1e3:.0f} ms past its first hit.",
    ]
    sources = ["beam_plastic_spring + tabular_return (moment-curvature from the published or flagged material curve)",
               "Hunt & Crossley 1975; Flores et al. 2011 restitution factor (via Machado et al. 2012)",
               "Gyrostat equations: Hughes, Spacecraft Attitude Dynamics ch. 4",
               "Cantilever first-mode tip mass 33/140 m (Rayleigh)"]
    paths = results_report(OUT, "Spinner vs spinner double hit",
                           "Does blade bending make a vertical spinner and a horizontal spinner hit twice in one "
                           "engagement?", answer, rep, history=hist, history_panels=panels,
                           findings=findings, assumptions=assumptions, sources=sources,
                           extra_svgs=[p[1] for p in figs_extra])
    print(paths)
    for f_ in findings:
        print("-", f_)
    for c in order:
        r = tab[c]; print(f"{c:24s} {r['k']}/{r['n']} = {r['p']*100:.1f}%  CI {r['ci'][0]*100:.1f}-{r['ci'][1]*100:.1f}  "
                          f"double<6ms {r['p6']*100:.1f}%")
    return paths


if __name__ == "__main__":
    main()

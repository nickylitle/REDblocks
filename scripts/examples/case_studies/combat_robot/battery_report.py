"""Report for the battery Monte Carlo (battery_study.py): minimum 6S pack for a 180 s fight.
Pages: answer + verification (results_report on a verified nominal fight), study figures (minimum pack vs weapon
use and pushing, pass rate by pack size, what drives it, where the energy goes, weapon ESC limit), model
schematic. Writes one merged PDF."""
import math
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from battery_life import run, DEFAULTS
from battery_study import DIMS, CUTS, OUT
from simblocks.viz import theme as T
from simblocks.viz.diagram import Schematic
from simblocks.viz.report import results_report
from simblocks.diagnostics.verify import SimSpec, verify_all
from simblocks.diagnostics.failure import FailureMode

PACKS = [450, 550, 650, 750, 850, 1000, 1300, 1500]
PRIMARY = "3.3 V/cell"


def nominal():
    p = {k: (v[0] + v[1]) / 2 for k, v in DIMS.items()}
    p["n_spinups"] = int(round(p["n_spinups"]))
    return p


def critical_dt(p):
    q = dict(DEFAULTS); q.update(p)
    J_w = 2.0 * (q["J_wheel"] + q["J_rot_d"] * q["G_d"] ** 2)
    return 2.0 * J_w * 0.02 / (max(q["F_push"], 1e-6) * q["r_wheel"] ** 2)


def spec(params, cap):
    fm = [FailureMode("charge used", metric="used_frac", limit=0.80, fails_when="above",
                      origin="user (accepted proposal): pack too small if a fight uses more than 80% of rated mAh"),
          FailureMode("voltage sag", metric="v_min_cell", limit=3.3, fails_when="below",
                      origin="user (accepted proposal): loaded voltage must stay at or above 3.3 V per cell")]
    unc = {k: (v[0], v[1]) for k, v in DIMS.items() if k not in ("n_spinups", "push_s", "duty_drive", "f_after_hit")}
    return SimSpec(
        name="Battery over a 180 s fight", run=lambda p, d: run(p, d), params={**params, "cap_mAh": cap}, dt=1e-4,
        metrics=["Wh", "mAh_used", "v_min_cell", "I_peak", "t_first_full"],
        uncertain=unc,
        knockouts={"no blade windage": (dict(windage_on=0.0), {"Wh": "-"}),
                   "no weapon hold (governor off)": (dict(hold_on=0.0), {"Wh": "-"}),
                   "no pushing": (dict(push_s=0.0), {"Wh": "-"}),
                   "no normal driving": (dict(duty_drive=0.0), {"Wh": "-"}),
                   "no hits (one spin-up only)": (dict(n_spinups=1), {"Wh": "-"}),
                   "no pack recovery circuit (R1 = 0)": (dict(R1_frac=0.0), {})},
        limits=[("idle only: 3 W x 180 s = 0.15 Wh",
                 dict(rpm_target=1e-3, hold_on=0.0, duty_drive=0.0, push_s=0.0),
                 {"Wh": lambda v: abs(v - 0.150) < 0.003})],
        bounds={"v_min_cell": (0.0, 4.25), "Wh": (0.0, 100.0), "I_peak": (0.0, 400.0)},
        critical_dt=critical_dt, mc_n=20, ledger_tol=0.05, failure_modes=fm)


def main():
    T.apply()
    mc = pickle.load(open(os.path.join(OUT, "mc.pkl"), "rb"))
    R = mc["R"]
    P = {k: np.array([r["params"][k] for r in R], float) for k in DIMS}
    cap = {c: np.array([r[c]["cap"] for r in R], float) for c in CUTS}
    gov = {c: [r[c]["governs"] for r in R] for c in CUTS}
    n = len(R)
    figs = []

    # ---------------- recommendation from the data (mine, labelled)
    q = {c: np.nanpercentile(cap[c], [50, 90, 95, 100]) for c in CUTS}
    pass_rate = {c: [float(np.mean(cap[c] <= pk)) for pk in PACKS] for c in CUTS}

    # ---------------- verify the nominal fight at the pack that covers 95% of fights under the primary cutoff
    p_nom = nominal()
    rec_pack = next((pk for pk in PACKS if np.mean(cap[PRIMARY] <= pk) >= 0.95), PACKS[-1])
    rep = verify_all(spec(p_nom, rec_pack))
    print(rep.summary())

    # ---------------- figure 1: minimum pack vs spin-ups, coloured by pushing time
    fig = plt.figure(figsize=(13, 6.6))
    top = T.header(fig, "Minimum 6S pack for a 180 s fight",
                   f"{n} Sobol-sampled fights; each dot is the smallest rated capacity that passes. Colour = pushing time",
                   tag="SIMBLOCKS // BATTERY MONTE CARLO")
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.90, bottom=0.17, top=top - 0.08, wspace=0.18)
    sc = None
    for j, c in enumerate(["capacity only", PRIMARY]):
        ax = fig.add_subplot(gs[0, j])
        ns = P["n_spinups"].round()
        sc = ax.scatter(ns + (np.random.default_rng(j).random(n) - 0.5) * 0.5, cap[c], c=P["push_s"], cmap="viridis",
                        s=12, alpha=0.8, lw=0, vmin=0, vmax=60, zorder=3)
        for k in np.unique(ns):
            m = ns == k
            if m.sum() >= 3:
                ax.plot([k - 0.35, k + 0.35], [np.nanpercentile(cap[c][m], 95)] * 2, color=T.INK, lw=1.6, zorder=4)
        for pk in (450, 650, 850):
            ax.plot([2, 20.4], [pk, pk], color=T.INK_3, lw=0.8, ls=(0, (3, 3)), zorder=2)
            ax.text(20.6, pk, f"{pk}", va="center", ha="left", fontsize=7.5, color=T.INK_3, family=T.MONO)
        ax.set_xlim(2, 21.8); ax.set_ylim(0, max(1600, np.nanmax(cap[PRIMARY]) * 1.05))
        ax.set_xlabel("weapon spin-ups per fight (incl. the first)")
        ax.set_title("charge only: use under 80% of rated mAh" if c == "capacity only"
                     else "charge and sag: also stay above 3.3 V/cell under load", fontsize=9.5)
        if j == 0:
            ax.set_ylabel("minimum rated capacity (mAh)")
        T.style_ticks(ax)
    cax = fig.add_axes([0.915, 0.17, 0.012, top - 0.25])
    cb = fig.colorbar(sc, cax=cax); cb.set_label("pushing time (s)", color=T.INK_2, fontsize=8)
    cb.ax.tick_params(colors=T.INK_2, labelsize=7)
    T.footer(fig, "White bars: 95th percentile of the minimum pack for that spin-up count. Dashed lines: common 6S packs (mAh).")
    figs.append(T.save(fig, os.path.join(OUT, "min_pack_vs_use")))

    # ---------------- figure 2: pass rate by pack size + what drives it
    fig = plt.figure(figsize=(13, 6.2))
    top = T.header(fig, "Which pack covers which share of fights",
                   "Share of the sampled fights each pack finishes, and what pushes the minimum pack up",
                   tag="SIMBLOCKS // BATTERY MONTE CARLO")
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, bottom=0.14, top=top - 0.08, wspace=0.45,
                          width_ratios=[1.0, 1.1])
    ax = fig.add_subplot(gs[0, 0])
    for i, c in enumerate(CUTS):
        ax.plot(PACKS, np.array(pass_rate[c]) * 100, marker="o", ms=4, lw=1.6, color=T.SERIES[i], label=c)
    ax.axhline(95, color=T.STATUS["WARN"], lw=0.9, ls=(0, (3, 3)))
    ax.set_xlabel("rated pack capacity (mAh)"); ax.set_ylabel("fights finished (%)"); ax.set_ylim(0, 103)
    ax.set_title("Share of fights the pack finishes", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8); T.style_ticks(ax)
    ax = fig.add_subplot(gs[0, 1])
    from scipy.stats import spearmanr
    rho = []
    for k in DIMS:
        m = np.isfinite(cap[PRIMARY])
        rho.append(spearmanr(P[k][m], cap[PRIMARY][m]).correlation)
    order = np.argsort(np.abs(rho))
    labs = [DIMS[k][2] for k in DIMS]
    ax.barh(np.arange(len(rho)), np.array(rho)[order], color=[T.SERIES[0] if v > 0 else T.SERIES[1]
                                                               for v in np.array(rho)[order]], height=0.6)
    ax.set_yticks(np.arange(len(rho))); ax.set_yticklabels([labs[i] for i in order], fontsize=7.5)
    ax.axvline(0, color=T.INK_3, lw=0.8); ax.set_xlim(-1, 1)
    ax.set_xlabel("rank correlation with minimum pack (3.3 V/cell)")
    ax.set_title("What drives the minimum pack", fontsize=9.5); T.style_ticks(ax)
    T.footer(fig, "Blue: more of this needs a bigger pack. Orange: more of this needs a smaller pack. Dashed: 95% of fights.")
    figs.append(T.save(fig, os.path.join(OUT, "pass_rate_and_drivers")))

    # ---------------- figure 3: where the energy goes (median-energy fight at 650 mAh) + ESC limit
    E = np.array([r["E"]["E_cell"] for r in R]); med = int(np.argsort(E)[len(E) // 2])
    e = R[med]["E"]
    parts = [("blade KE lost in hits", e["E_hits"]), ("windage", e["E_windage"]),
             ("weapon motor no-load", e["E_weapon_noload"]), ("weapon copper", e["E_weapon_cu"]),
             ("friction drive + gearbox", e["E_transmission"]), ("hub + tread drag", e["E_bearing"]),
             ("normal driving", e["E_drive_in"]), ("pushing: tire slip", e["E_tire_slip"]),
             ("pushing: drive copper", e["E_push_cu"]), ("pack resistance", e["E_Rloss"]),
             ("electronics idle", e["E_quiescent"]), ("blade KE left at the end", e["dKE_blade"])]
    fig = plt.figure(figsize=(13, 6.0))
    top = T.header(fig, "Where the energy goes, and the weapon ESC limit",
                   f"Median-energy sampled fight on a 650 mAh pack: {e['E_cell']/3600:.1f} Wh from the cells",
                   tag="SIMBLOCKS // BATTERY MONTE CARLO")
    gs = fig.add_gridspec(1, 2, left=0.17, right=0.98, bottom=0.15, top=top - 0.08, wspace=0.42, width_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0, 0])
    vals = np.array([v for _, v in parts]) / 3600
    ax.barh(np.arange(len(parts)), vals, color=T.SERIES[0], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + vals.max() * 0.01, i, f"{v:.2f} Wh  ({v / vals.sum() * 100:.0f}%)", va="center", fontsize=7.5,
                color=T.INK_2, family=T.MONO)
    ax.set_yticks(np.arange(len(parts))); ax.set_yticklabels([n_ for n_, _ in parts], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, vals.max() * 1.45); ax.set_xlabel("energy (Wh)")
    ax.set_title("Energy by destination", fontsize=9.5); T.style_ticks(ax)
    ax = fig.add_subplot(gs[0, 1])
    lims = [25, 30, 35, 40, 45]
    tsp = [run({**p_nom, "I_lim_w": L, "cap_mAh": 650.0, "n_spinups": 1, "duty_drive": 0.0, "push_s": 0.0})["t_first_full"]
           for L in lims]
    ax.plot(lims, tsp, marker="o", color=T.SERIES[1], lw=1.6)
    for L, tv in zip(lims, tsp):
        ax.text(L - 0.4, tv - 0.06, f"{tv:.2f} s", ha="right", va="top", fontsize=7.5, color=T.INK_2, family=T.MONO)
    ax.set_xlabel("weapon ESC current limit, per motor (A)"); ax.set_ylabel("spin-up from rest to 12,000 rpm (s)")
    ax.set_title("Spin-up time vs ESC limit (650 mAh, fresh pack)", fontsize=9.5); ax.set_xlim(21, 47)
    ax.set_ylim(0, max(tsp) * 1.25); T.style_ticks(ax)
    T.footer(fig, "Spin-up is current-limited the whole way: back-EMF at 12,000 rpm blade speed is about half the pack voltage.")
    figs.append(T.save(fig, os.path.join(OUT, "energy_and_esc")))

    # ---------------- schematic
    sch = Schematic("Battery fight model", "One 6S pack feeding two weapon ESCs and two drive ESCs over 180 s")
    sch.body("pack", "mass", "6S pack", info=["OCV(SOC) + R0 + R1||C1", "R0 scales with 1/capacity"])
    sch.body("wesc", "motor", "Weapon ESCs x2", info=["2807 1350 KV", "current limit 25-45 A"])
    sch.body("blade", "rotor", "Blade", info=["J 7.6-8.8e-4 kg m^2", "12,000 rpm target"])
    sch.body("desc", "motor", "Drive ESCs x2", info=["Compact V2 (1806 2300 KV)", "20 A limit"])
    sch.body("floor", "ground", "")
    sch.link("pack", "wesc", "rigid", "Bus", "constant-power load", "ecm_constant_power")
    sch.link("wesc", "blade", "spring", "Friction drive", "G 1.25-1.5, eta 0.75-0.95", "esc_supply")
    sch.link("pack", "desc", "rigid", "Bus", "drive power trace / push", "motor_current")
    sch.link("desc", "floor", "spring", "Tires", "push traction 8-150 N", "coulomb_tanh")
    sch.probe("pack", "charge used, loaded voltage")
    figs.append(sch.draw(os.path.join(OUT, "battery_schematic")))

    # ---------------- findings
    rows = []
    for c in CUTS:
        rows.append(f"{c}: minimum pack median {q[c][0]:.0f} mAh, 90th pct {q[c][1]:.0f}, 95th {q[c][2]:.0f}, "
                    f"worst sampled {q[c][3]:.0f}; governs: "
                    f"{sum(1 for g in gov[c] if g == 'capacity')} charge / {sum(1 for g in gov[c] if g == 'sag')} sag")
    findings = rows + [
        "Pass rate by pack (3.3 V/cell): " + ", ".join(f"{pk} mAh {pr*100:.0f}%" for pk, pr in zip(PACKS, pass_rate[PRIMARY])),
        "Pass rate by pack (3.0 V/cell): " + ", ".join(f"{pk} mAh {pr*100:.0f}%" for pk, pr in zip(PACKS, pass_rate["3.0 V/cell"])),
        "Pass rate by pack (charge only): " + ", ".join(f"{pk} mAh {pr*100:.0f}%" for pk, pr in zip(PACKS, pass_rate["capacity only"])),
        f"Smallest common pack finishing at least 95% of sampled fights at 3.3 V/cell: {rec_pack} mAh.",
    ]
    assumptions = [
        "Weapon: two 2807 1350 KV motors (R 45-58 mOhm line-line from Axisflying, Hobbywing and GEPRC data; no-load "
        "current 1-2 A), each on its own current-limited ESC, friction-driving the blade at ratio 1.25-1.5.",
        "Blade held at 12,000 rpm by throttle after each spin-up; windage from strip theory with an edge Cd of "
        "0.05-0.4 (the air co-rotates with a bar spinning in its own plane, so the flat-plate 1.28 overstates it).",
        "Normal driving replays the electrical power of the legacy drive sim's aggressive 30 s segment at 6S (55 W "
        "average while driving). Pushing: robot blocked, wheels slip at the traction limit or stall at the 20 A limit.",
        "Pack: 6 cells, resistance per 450 mAh cell 8-25 mOhm scaled with 1/capacity, R1 = 0.5 R0, 20 s recovery; "
        "generic LiPo OCV curve (3.3-4.2 V). HV packs (4.35 V) would add about 3% energy.",
        "Drive and weapon on the same 6S pack. Your notes say the drive ran on 5S before.",
    ]
    sources = ["Plett 2015 (one-RC pack model)", "Motor data: Axisflying AZ2807, Hobbywing XRotor 2807, GEPRC EM2807",
               "Strip-theory windage", "Legacy Flapjack drive sim (drive power trace)"]
    answer = dict(label="Minimum 6S pack (95% of fights)", value=f"{rec_pack}", unit="mAh",
                  status="PASS" if rep.ok else "WARN",
                  sub=f"at 3.3 V/cell; median fight {q[PRIMARY][0]:.0f} mAh")
    paths = results_report(OUT, "Battery for a 180 s fight",
                           "What is the smallest 6S pack that finishes a full fight?", answer, rep,
                           findings=findings, assumptions=assumptions, sources=sources,
                           extra_svgs=[f[1] for f in figs if isinstance(f, (list, tuple)) and len(f) > 1])
    # merged PDF: results_report pages + study figures
    import pypdf
    from matplotlib.backends.backend_pdf import PdfPages
    extra_pdf = os.path.join(OUT, "_extra.pdf")
    with PdfPages(extra_pdf) as pdf:
        for f in figs:
            png = f[0] if isinstance(f, (list, tuple)) else f
            img = plt.imread(png)
            fg = plt.figure(figsize=(img.shape[1] / 200, img.shape[0] / 200), dpi=200)
            fg.figimage(img, 0, 0); pdf.savefig(fg, facecolor=T.SURFACE); plt.close(fg)
    w = pypdf.PdfWriter()
    rp = pypdf.PdfReader(paths["pdf"]); ep = pypdf.PdfReader(extra_pdf)
    w.add_page(rp.pages[0])
    for pg in ep.pages:
        w.add_page(pg)
    for pg in rp.pages[1:]:
        w.add_page(pg)
    final = os.path.join(OUT, "flapjack_battery_report.pdf")
    w.write(final)
    print(final)
    for f_ in findings:
        print("-", f_)
    return final, rep, q, pass_rate, rec_pack


if __name__ == "__main__":
    main()

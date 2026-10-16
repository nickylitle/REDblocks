"""End-to-end demo of the v0.2 output stack on the horizontal wedge strike:
schematic + signal-flow diagrams, user-defined failure modes, verification, results report (PNG/SVG/PDF/HTML)."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wedge_strike_from_blocks import wedge_strike, critical_dt, build, DEFAULTS
from simblocks.materials import get
from simblocks.geometry.contact_geometry import first_contact_delta, first_contact_stats, sample_first_contact
from simblocks.structures.impact import shear_plug_capacity, punch_perimeter
from simblocks.diagnostics.verify import SimSpec, verify_all
from simblocks.diagnostics.failure import FailureMode
from simblocks.viz.diagram import Schematic, SignalFlow
from simblocks.viz.report import results_report

GEOMS = {"sim": dict(wedge_l=0.030, wedge_angle0_deg=60.0, label="sim geometry (30 mm, 60 deg)"),
         "cad": dict(wedge_l=0.037, wedge_angle0_deg=58.8, label="CAD geometry (37 mm slant, 58.8 deg)")}
GEO = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in GEOMS else "cad"
G_ = {k: v for k, v in GEOMS[GEO].items() if k != "label"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"wedge_v02_{GEO}"); os.makedirs(OUT, exist_ok=True)
TOOTH_W = 6.35e-3          # ASSUMPTION: tooth contact width = blade thickness (user has not given one)
KAPPA = 2.2                # 3-edge-clamped CAD reading (flapjack-wedge-final-model.md)
T_DESIGN = 0.0025
rpm = 1257 * 60 / (2 * math.pi); D_blade = 0.18
delta = first_contact_delta(3.0, rpm, D_blade); fc = first_contact_stats(delta)
AMAX, AMED = math.degrees(fc["alpha_max"]), math.degrees(fc["median"])
NOM = dict(t=T_DESIGN, physics="curve", edge_clamp_factor=KAPPA, weapon_angle0_deg=-AMAX, **G_)   # worst-allowable hit


def run(p, dt):
    out = wedge_strike(p, dt)
    q = dict(DEFAULTS); q.update(p)
    su = get(q["material"]).su * q["rate_factor"]
    out["plug_capacity_N"] = shear_plug_capacity(punch_perimeter(w=TOOTH_W, l=q["t"]), q["t"], su)
    return out


modes = [
    FailureMode("tip drop", metric="tip_drop_perm_mm", limit=5.0, fails_when="above",
                origin="user: wedge fails at a permanent 5 mm tip drop (tip fall from face rotation)",
                reach_metric="max_possible_bend_drop_mm"),
    FailureMode("puncture", check=lambda o: (o["peak_dent_force_N"] >= o["plug_capacity_N"],
                                             o["plug_capacity_N"] / max(o["peak_dent_force_N"], 1e-9)),
                origin="user: wedge fails if punctured (shear-plug capacity vs peak tooth force)",
                assumption="tooth contact patch 6.35 mm x t"),
]

spec = SimSpec(
    name=f"Wedge 2.5 mm, horizontal strike, {GEO} geometry", run=run, params=NOM, dt=1e-7,
    metrics=["tip_drop_perm_mm", "bending deformation", "denting deformation", "peak_dent_force_N"],
    uncertain={"weapon_angle0_deg": lambda rng: -math.degrees(sample_first_contact(delta, 1, rng)[0]),
               "weapon_w": (1130.0, 1380.0), "rate_factor": (1.1, 1.3), "hinge_len_over_t": (0.5, 2.0),
               "edge_clamp_factor": (0.9 * KAPPA, 1.1 * KAPPA)},
    knockouts={"no dent plasticity": (dict(dent_on=0.0), {"denting deformation": "-"}),
               "no bend plasticity": (dict(bend_on=0.0), {"bending deformation": "-"})},
    limits=[("weapon stopped -> no damage", dict(weapon_w=1e-9), {"tip_drop_perm_mm": 0.0})],
    bounds={"tip_drop_perm_mm": (-1.0, 40.0), "bending deformation": (0.0, 90.0)},
    critical_dt=critical_dt, mc_n=400, failure_modes=modes)

rep = verify_all(spec)
nominal, hist, b = wedge_strike(NOM, 1e-7, history=True)
nominal_full = run(NOM, 1e-7)

# ---------------------------------------------------------------- diagrams (numbers come from build())
p = dict(DEFAULTS); p.update(NOM)
sch = Schematic(f"Wedge strike: physical model, {GEOMS[GEO]['label']}", "Horizontal spinner tooth hitting a hinged wedge face; values computed by the blocks shown")
sch.body("weapon", "rotor", "Weapon (rotor)", info=[f"I = {p['weapon_moi']:.3g} kg m^2", f"w0 = {p['weapon_w']:.0f} rad/s",
                                                  f"R = {p['weapon_radius']*1e3:.0f} mm", f"E = {0.5*p['weapon_moi']*p['weapon_w']**2:.0f} J"])
sch.body("lip", "point", "")
sch.body("face", "mass", "Wedge face", info=[f"I = {b['wedge_moi']:.2e} kg m^2", f"L = {p['wedge_l']*1e3:.0f} mm, {p['wedge_angle0_deg']:.0f} deg",
                                           f"t = {p['t']*1e3:.1f} mm 4130 @45 HRC"])
sch.body("bot", "mass", "Chassis", info=[f"m = {p['bot_mass']:.2f} kg"])
sch.body("gnd", "ground", "")
sch.link("weapon", "lip", "contact", "Tooth contact", "unilateral", "tip_face_penetration", lane=0.0)
sch.link("lip", "face", "elastoplastic", "Lip dent", f"k = {b['dent_k']:.3g} N/m\nFy = {b['dent_tab_F'][0]:.0f} N + depth hardening",
         "circ_clamped_center_point\n+ tab_shear_out_force", lane=0.0)
sch.link("face", "bot", "elastoplastic", "Root fold (hinge)", f"k = {b['bend_k']:.0f} N m/rad\nMp = {b['Mp_root']:.0f} N m (kappa {KAPPA})",
         "moment_curvature_rect -> hinge_table", lane=0.0)
sch.link("bot", "gnd", "rigid", "Free in x", "translation only", "")
sch.probe("face", "tip drop / puncture")
sch.note(f"worst-allowable hit alpha_max {AMAX:.1f} deg (delta {delta:.3f}, 3 m/s closing)")
sch.note("material law: " + b["curve_source"][:70])
sch_paths = sch.draw(os.path.join(OUT, "schematic"))

sf = SignalFlow("Wedge strike: per-step computation", "Semi-implicit Euler, dt = 0.1 us, njit kernel loop")
sf.stage("state", "State x(t)", ["weapon angle, w", "face angle, w_face", "chassis x, v", "dent set, bend set"], "state")
sf.stage("geom", "Geometry", [("tip_on_circle", "tooth tip position"), ("tip_face_penetration", "pen + energy arm")], "geometry")
sf.stage("const", "Constitutive", [("tabular_return (dent)", "one-sided, gouge hardening"), ("tabular_return (hinge)", "M-theta from curve")], "constitutive")
sf.stage("force", "Forces", [("dent force -> weapon torque", ""), ("root moment -> face", ""), ("reaction -> chassis", "")], "force")
sf.stage("integ", "Integrate", [("semi-implicit Euler", "v then x"), ("energy ledger", "plastic work, KE")], "integrate")
sf.stage("out", "Outputs", [("tip_drop_perm_mm", "failure: tip drop"), ("peak_dent_force_N", "failure: puncture")], "output")
for a, c, l in (("state", "geom", "angles"), ("geom", "const", "pen, theta"), ("const", "force", "F, M"),
                ("force", "integ", "accel"), ("integ", "out", "")):
    sf.edge(a, c, l)
sf.feedback("integ", "state", "x(t + dt)")
sfl_paths = sf.draw(os.path.join(OUT, "signal_flow"))

# ---------------------------------------------------------------- design sweep: thickness at the worst-allowable hit
ts = np.array([1.5, 2.0, 2.5, 3.0, 3.5]) * 1e-3
sweep_sim = [wedge_strike(dict(NOM, t=t, wedge_l=0.030, wedge_angle0_deg=60.0))["tip_drop_perm_mm"] for t in ts]
sweep_cad = [wedge_strike(dict(NOM, t=t, wedge_l=0.037, wedge_angle0_deg=58.8))["tip_drop_perm_mm"] for t in ts]

fm = rep.failure
td, pu = fm["tip drop"], fm["puncture"]
findings = [
    f"Worst-allowable hit (alpha_max {AMAX:.1f} deg): permanent tip drop {nominal_full['tip_drop_perm_mm']:.2f} mm vs 5 mm limit (SF {td['nominal_SF']:.2f}).",
    f"Across {td['n']} Monte Carlo hits: P(tip drop >= 5 mm) = {td['p_fail']*100:.2g}% "
    f"[95% CI {td['p_fail_ci95'][0]*100:.2g}-{td['p_fail_ci95'][1]*100:.2g}%], lowest SF {td['SF_min']:.2f}.",
    f"Puncture margin SF {pu['nominal_SF']:.2f} at the worst hit (peak tooth force {nominal_full['peak_dent_force_N']/1e3:.1f} kN vs "
    f"plug capacity {nominal_full['plug_capacity_N']/1e3:.1f} kN); P(puncture) = {pu['p_fail']*100:.2g}%.",
    f"Governing mode: {next(k for k, v in fm.items() if v['governing'])}. Contact angle drives the margin "
    f"(rank corr {td['drivers'][0][1]:+.2f}); thickness is a weak lever across 1.5-3.5 mm.",
    f"Geometry limit: the face folding to vertical drops the tip at most {nominal_full['max_possible_bend_drop_mm']:.2f} mm "
    f"(L - L sin a0); the 5 mm limit is {'reachable' if nominal_full['max_possible_bend_drop_mm'] >= 5 else 'UNREACHABLE'} here.",
    f"Lip dent {nominal_full['denting deformation']:.2f} mm is reported separately (tip drop with dent included: "
    f"{nominal_full['tip_drop_with_dent_mm']:.2f} mm).",
]
assumptions = [
    "Root capacity kappa = 2.2 (3-edge-clamped CAD reading, flapjack-wedge-final-model.md), swept +/-10%.",
    "Tooth contact patch 6.35 mm (blade thickness) x t for puncture: PLACEHOLDER, user value pending.",
    "Plastic hinge length = t, swept 0.5-2 t. Rate factor 1.1-1.3 on yield.",
    "4130 @45 HRC curve: Ramberg-Osgood fallback (published fit pending).",
    "LEGACY block in use: tab_shear_out_force (dent depth hardening, from flapjack-wedge-final-model); published replacement pending.",
    "Contact angle sampled from the first-contact distribution, 3 m/s closing, 12,000 rpm, 180 mm blade.",
]
sources = ["Roark 7e Table 11.2 (dent stiffness)", "Roark 7e Table 8.1 (root flexure)", "yield-line theory (root fold, lip fan)",
           "blade-angle-first-contact.md (contact-angle distribution)", "flapjack-wedge-final-model.md (kappa, gouge hardening)",
           "Horizontal Wedge Impact.py (geometry, coupling)"]
unreach = [k for k, v in fm.items() if v.get("reach") and not v["reach"]["reachable"]]
failed = any(v["n_failed"] or v["nominal_failed"] for v in fm.values())
gmax = nominal_full["max_possible_bend_drop_mm"]
if failed:
    answer = dict(label="Is 2.5 mm OK?", value="NO", unit="", status="FAIL",
                  sub=f"worst-allowable tip drop {nominal_full['tip_drop_perm_mm']:.2f} / 5 mm")
elif unreach:
    answer = dict(label="Is 2.5 mm OK?", value="N/A", unit="", status="WARN",
                  sub=f"5 mm drop unreachable: geometry max {gmax:.2f} mm")
else:
    answer = dict(label="Is 2.5 mm OK?", value="YES", unit="", status="PASS",
                  sub=f"worst tip drop {nominal_full['tip_drop_perm_mm']:.2f} / 5 mm (geometry max {gmax:.2f})")
panels = [dict(title="Permanent set at the lip", keys=["dent_set"], scale=1e3, unit="mm"),
          dict(title="Root hinge rotation set", keys=["bend_set"], scale=180 / math.pi, unit="deg", color=1),
          dict(title="Tooth force", keys=["dent_F"], scale=1e-3, unit="kN", color=4),
          dict(title="Root moment", keys=["root_M"], scale=1.0, unit="N m", color=2),
          dict(title="Weapon energy", keys=["weapon_KE"], scale=1.0, unit="J", color=3),
          dict(title="Plastic work absorbed", keys=["W_plastic"], scale=1.0, unit="J", color=0)]
sweep = dict(x=ts, x_scale=1e3, x_label="wedge thickness (mm)", y_label="permanent tip drop (mm)",
             title="Tip drop vs thickness at the worst-allowable hit", limit=5.0, limit_label="5 mm FAIL",
             series={"sim geometry (30 mm, 60 deg)": sweep_sim, "CAD geometry (37 mm, 58.8 deg)": sweep_cad},
             mark=T_DESIGN,
             input_labels={"weapon_angle0_deg": "contact angle (shallower = +)", "hinge_len_over_t": "hinge length / t",
                           "weapon_w": "weapon speed", "rate_factor": "strain-rate factor", "edge_clamp_factor": "edge clamp kappa"})
paths = results_report(OUT, f"Wedge 2.5 mm, horizontal strike, {GEO.upper()} geometry", "Does a 2.5 mm 4130 wedge survive a worst-allowable 600 J horizontal hit?",
                       answer, rep, history=hist, history_panels=panels, sweep=sweep, findings=findings,
                       assumptions=assumptions, sources=sources, extra_svgs=[sch_paths[1], sfl_paths[1]])
print(rep.summary())
for f in findings:
    print("-", f)
print(paths)

"""TRIAL 3: carbon top plate carrying a motor, unit dropped flat.
Prompt (as a user would give it): "My motor (120 g) hangs from the middle of a 160 x 120 mm top plate of 3K
twill [0/45/0/45]s, screwed down on all four edges. If the unit is dropped and lands flat from 1 m, does the plate
crack (first-ply failure) or flex more than 5 mm into the rotor below it?"
Blocks: laminate_card (CLT) -> as_isotropic(flexural) -> rect_ss_point_at (Navier, patch) and rect_ss_center_point
(Roark centre moment with r0 correction) -> first_ply_failure (Tsai-Wu); penalty_kelvin_voigt +
damping_for_restitution (floor), critical_dt_spring."""
import math, os, sys
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from simblocks.materials.cards import laminate_card
from simblocks.materials import clt
from simblocks.structures.plates.rectangular import rect_ss_point_at, rect_ss_center_point
from simblocks.contact.penalty import penalty_kelvin_voigt, damping_for_restitution
from simblocks.integrators.explicit import critical_dt_spring
from simblocks.diagnostics.verify import SimSpec, verify_all
from simblocks.diagnostics.failure import FailureMode
from simblocks.viz.report import results_report

G0 = 9.81
DEFAULTS = dict(layup="CF_3K_twill [0/45/0/45]s", a=0.16, b=0.12, patch=0.028, m_total=1.361, m_mot=0.12, h=1.0,
                k_floor=2.0e5, e_floor=0.3, g=G0, damp_on=1.0, plate_rigid=0.0, t_end=0.04)


def build(p):
    card = laminate_card(p["layup"])
    abd = card.abd
    iso = card.as_isotropic("flexural")
    t = abd["h"]
    kp = rect_ss_point_at(iso, p["a"], p["b"], t, p["a"] / 2, p["b"] / 2, patch=(p["patch"], p["patch"]))["k"]
    if p["plate_rigid"] > 0.5:
        kp *= 1e4
    r0 = p["patch"] / math.sqrt(math.pi)                     # equal-area radius of the square motor flange
    s_per_N = rect_ss_center_point(iso, p["a"], p["b"], t, r0=r0)["stress_per_N"]
    m_per_N = s_per_N * t * t / 6.0                         # centre bending moment per unit width per newton
    # ASSUMPTION: near-square plate, Mx = My at the centre (both directions loaded equally)
    R, ply, ang = clt.first_ply_failure(abd, M=(m_per_N, m_per_N, 0.0))
    mc = p["m_total"] - p["m_mot"]
    c_floor = damping_for_restitution(p["e_floor"], p["k_floor"], p["m_total"]) * p["damp_on"]
    return dict(kp=kp, F_fpf=R, ply=ply, ang=ang, t=t, mc=mc, c_floor=c_floor, E_flex=iso.E, card=card)


@njit(cache=True)
def _loop(n, dt, mc, mm, kp, kf, cf, g, v0, every, H):
    # y positive DOWN; chassis bottom touches the floor at yc = 0 at t = 0; motor starts at static sag on the plate
    yc = 0.0; vc = v0; d0 = mm * g / kp; ym = d0; vm = v0
    E_damp = 0.0; E_grav = 0.0; E_pl = 0.0; Fpk = 0.0; dmax = 0.0; Ffpk = 0.0; acpk = 0.0; j = 0
    for i in range(n):
        Ff = penalty_kelvin_voigt(yc, vc, kf, cf)
        Fp = kp * (ym - yc)                                  # plate force on the motor (up) and chassis (down)
        ac = (mc * g + Fp - Ff) / mc
        am = (mm * g - Fp) / mm
        vc += ac * dt; vm += am * dt
        # work bookkeeping with the same (updated) velocities the positions use
        E_damp += Ff * vc * dt                               # energy taken by the floor (spring + dashpot)
        E_grav += (mc * vc + mm * vm) * g * dt               # work done by gravity
        E_pl += Fp * (vm - vc) * dt                          # energy into the plate spring
        yc += vc * dt; ym += vm * dt
        if abs(Fp) > Fpk:
            Fpk = abs(Fp)
        if abs(ym - yc) > dmax:
            dmax = abs(ym - yc)
        if Ff > Ffpk:
            Ffpk = Ff
        if abs(ac) > acpk:
            acpk = abs(ac)
        if every > 0 and i % every == 0 and j < H.shape[0]:
            H[j, 0] = i * dt; H[j, 1] = Fp; H[j, 2] = ym - yc; H[j, 3] = Ff; H[j, 4] = yc
            j += 1
    E0 = 0.5 * (mc + mm) * v0 * v0
    E1 = 0.5 * mc * vc * vc + 0.5 * mm * vm * vm
    # kinetic ledger: KE0 + gravity work - floor energy - plate energy change = KE1, and the plate spring's
    # integrated power must equal its stored-energy change (checks the coupling, not just the integrator)
    Eplate = 0.5 * kp * ((ym - yc) ** 2 - d0 * d0)
    res = (E0 + E_grav - E_damp - E_pl - E1) + (E_pl - Eplate)
    return Fpk, dmax, Ffpk, acpk, res / max(E0, 1e-12), j, E_damp


def run(params=None, dt=2e-6, history=False):
    p = dict(DEFAULTS); p.update(params or {}); b = build(p)
    n = int(round(p["t_end"] / dt)); every = max(n // 800, 1) if history else 0
    H = np.zeros((n // max(every, 1) + 2 if history else 1, 5))
    v0 = math.sqrt(2 * p["g"] * p["h"])
    Fp, d, Ff, ac, led, j, Efl = _loop(n, dt, b["mc"], p["m_mot"], b["kp"], p["k_floor"], b["c_floor"], p["g"], v0,
                                  every, H)
    out = dict(F_plate_peak_N=Fp, plate_defl_mm=d * 1e3, F_floor_peak_N=Ff, chassis_g=ac / G0,
               F_fpf_N=float(b["F_fpf"]), fpf_SF=float(b["F_fpf"]) / max(Fp, 1e-9), E_floor_J=Efl, _ledger_rel=led)
    if history:
        Hh = H[:j]
        return out, dict(t=Hh[:, 0], Fp=Hh[:, 1], defl=Hh[:, 2] * 1e3, Ff=Hh[:, 3], yc=Hh[:, 4] * 1e3)
    return out


def critical_dt(params):
    p = dict(DEFAULTS); p.update(params); b = build(p)
    return min(critical_dt_spring(b["kp"], p["m_mot"]), critical_dt_spring(p["k_floor"], b["mc"]))


modes = [
    FailureMode("plate cracks", check=lambda o: (o["F_plate_peak_N"] >= o["F_fpf_N"], o["fpf_SF"]),
                origin="trial user: plate fails at first-ply failure under the motor"),
    FailureMode("plate hits rotor", metric="plate_defl_mm", limit=5.0, fails_when="above",
                origin="trial user: plate may not flex more than 5 mm (rotor clearance)", assumption="5 mm clearance"),
]
SPEC = SimSpec(
    name="Trial 3 CF top plate drop", run=lambda p, dt: run(p, dt), params={}, dt=2e-6,
    metrics=["F_plate_peak_N", "plate_defl_mm", "F_floor_peak_N"],
    uncertain={"h": (0.5, 1.5), "k_floor": (1.0e5, 4.0e5), "e_floor": (0.2, 0.5), "m_mot": (0.10, 0.14)},
    knockouts={"rigid plate": (dict(plate_rigid=1.0), {"plate_defl_mm": "-"}),
               "no floor damping": (dict(damp_on=0.0), {"F_floor_peak_N": "any"})},
    limits=[("no gravity, no drop -> no load", dict(h=0.0, g=0.0), {"F_plate_peak_N": 0.0})],
    bounds={"plate_defl_mm": (0.0, 50.0), "F_plate_peak_N": (0.0, 1e5)},
    critical_dt=critical_dt, mc_n=200, failure_modes=modes)

if __name__ == "__main__":
    OUT = os.path.join(HERE, "out", "trial3"); os.makedirs(OUT, exist_ok=True)
    rep = verify_all(SPEC)
    nom, hist = run({}, 2e-6, history=True)
    b = build(dict(DEFAULTS))
    lays = {"[0/45]s (4 plies)": "CF_3K_twill [0/45]s", "[0/45/0]s (6 plies)": "CF_3K_twill [0/45/0]s",
            "[0/45/0/45]s (8 plies)": "CF_3K_twill [0/45/0/45]s", "[0/45/0/45/0]s (10 plies)": "CF_3K_twill [0/45/0/45/0]s"}
    hs = [0.5, 1.0, 1.5, 2.0]
    series = {"8 plies, first-ply SF": [run(dict(h=h))["fpf_SF"] for h in hs],
              "6 plies, first-ply SF": [run(dict(h=h, layup=lays["[0/45/0]s (6 plies)"]))["fpf_SF"] for h in hs]}
    sweep = dict(x=np.array(hs), x_scale=1.0, x_label="drop height (m)", y_label="first-ply safety factor",
                 title="Plate crack margin vs drop height", limit=1.0, limit_label="SF 1 = cracks", fails_when="below",
                 series=series, mark=1.0, input_labels={"h": "drop height", "k_floor": "wheel/floor stiffness",
                                                        "e_floor": "landing restitution", "m_mot": "motor mass"})
    fm = rep.failure
    failed = any(v["nominal_failed"] for v in fm.values())
    answer = dict(label="Plate survives a 1 m flat landing?", value="NO" if failed else "YES", unit="",
                  status="FAIL" if failed else "PASS",
                  sub=f"first-ply SF {nom['fpf_SF']:.2f}, flex {nom['plate_defl_mm']:.2f} mm")
    findings = [
        f"{b['card'].name}: {b['t']*1e3:.2f} mm thick, flexural E {b['E_flex']/1e9:.1f} GPa, plate stiffness under the "
        f"motor {b['kp']/1e3:.0f} kN/m.",
        f"1 m flat landing: chassis peaks at {nom['chassis_g']:.0f} g, plate carries {nom['F_plate_peak_N']:.0f} N "
        f"from the motor and flexes {nom['plate_defl_mm']:.2f} mm.",
        f"First-ply failure (Tsai-Wu) at {b['F_fpf']:.0f} N under the motor flange, ply {b['ply']} at "
        f"{b['ang']:.0f} deg: SF {nom['fpf_SF']:.2f}.",
        f"Monte Carlo ({fm['plate cracks']['n']} landings): P(crack) = {fm['plate cracks']['p_fail']*100:.0f}%, "
        f"P(> 5 mm flex) = {fm['plate hits rotor']['p_fail']*100:.0f}%. Top driver: "
        f"{sweep['input_labels'].get(fm['plate cracks']['drivers'][0][0], fm['plate cracks']['drivers'][0][0])} ({fm['plate cracks']['drivers'][0][1]:+.2f}).",
    ]
    assumptions = ["Plate simply supported on all four edges (screwed down; real screws give partial fixity).",
                   "Motor flange 28 mm square spreads the load (Navier patch for stiffness, Roark r0 for moment).",
                   "Centre moments Mx = My for first-ply failure (near-square plate, conservative for twill).",
                   "Laminate smeared to an equivalent isotropic plate (as_isotropic, flexural) for stiffness.",
                   "Flat landing on both wheels: floor + tyre stiffness 200 kN/m, restitution 0.3 (ASSUMED).",
                   "Plate mass and the plate's own vibration modes are neglected (motor dominates).",
                   "3K twill values: Rock West vendor CLT reference values, not design allowables."]
    sources = ["CLT (Jones; Daniel & Ishai), Tsai-Wu first-ply failure", "Navier series (exact SS plate)",
               "Roark 7e Table 11.4 case 1 with r0 correction", "Kawaguchi 1992 (restitution damping)"]
    panels = [dict(title="Plate force from the motor", keys=["Fp"], unit="N", t_scale=1e3, t_label="time (ms)",
                   limit=b["F_fpf"], limit_label="first-ply failure"),
              dict(title="Plate flex under the motor", keys=["defl"], unit="mm", color=1, t_scale=1e3,
                   t_label="time (ms)"),
              dict(title="Floor force on the wheels", keys=["Ff"], unit="N", color=2, t_scale=1e3, t_label="time (ms)")]
    paths = results_report(OUT, "Trial 3: CF top plate drop", "Does the 8-ply 3K twill top plate crack or flex into the "
                           "rotor when the unit lands flat from 1 m?", answer, rep, history=hist, history_panels=panels,
                           sweep=sweep, findings=findings, assumptions=assumptions, sources=sources)
    b["card"].render(os.path.join(OUT, "material_card"))
    print(rep.summary()); [print("-", f) for f in findings]

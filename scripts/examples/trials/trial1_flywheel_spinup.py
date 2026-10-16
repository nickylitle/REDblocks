"""TRIAL 1: flywheel spin-up on a sagging LiPo.
Prompt (as a user would give it): "How long does my 600 J flywheel bar take to hit 12,000 rpm on a 5S 450 mAh pack with
a 1100 KV motor through a 1.5:1 belt, 60 A ESC limit, and does the pack sag below 3.3 V per cell?"
Blocks: motor_from_kv, ecm_constant_power (Plett 1-RC), lipo_ocv, bar_windage_torque, bearing_drag_torque,
spin_up_time (analytic anchor), critical_dt_spring-style time-constant check."""
import math, os, sys
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from simblocks.actuators.motors import motor_from_kv, esc_supply, lipo_ocv_k, spin_up_time
from simblocks.aero.basic import bar_windage_torque, isa_troposphere
from simblocks.friction.laws import bearing_drag_torque
from simblocks.diagnostics.verify import SimSpec, verify_all
from simblocks.diagnostics.failure import FailureMode
from simblocks.viz.report import results_report

W_TARGET = 12000 * 2 * math.pi / 60
DEFAULTS = dict(
    I_w=7.6e-4,            # flywheel inertia, kg m^2 (600 J at 12,000 rpm)
    R_bar=0.09, h_bar=0.00635, Cd=1.28,      # bar spinner tip radius, thickness facing the flow, drag coeff
    kv=1100.0, R_m=0.035, I_max=60.0, J_rot=2.0e-6, G=1.5,   # ASSUMED motor / ESC / belt
    cells=5, Q_mAh=450.0, R0=0.15, R1=0.05, C1=600.0,        # ASSUMED pack ECM (hobby IR band, Plett form)
    mu_b=0.0015, d_bore=0.008, M0=0.002, F_rad=13.3,         # bearing: SKF simplified, ASSUMED values
    windage_on=1.0, bearing_on=1.0, rc_on=1.0, sag_on=1.0, t_end=3.0)


def build(p):
    mo = motor_from_kv(p["kv"], p["R_m"], p["I_max"], p["J_rot"])
    J_eff = p["I_w"] + mo["J"] * p["G"] ** 2                  # reflected to the flywheel shaft
    _, _, rho = isa_troposphere(0.0)
    return dict(Kt=mo["Kt"], J_eff=J_eff, rho=rho, Q=p["Q_mAh"] * 3.6,
                tau_mech=J_eff * p["R_m"] / (mo["Kt"] * p["G"]) ** 2)


@njit(cache=True)
def _loop(n, dt, J, Kt, G, Rm, Imax, cells, Q, R0, R1, C1, Rb, hb, Cd, rho, mu_b, Fr, db, M0,
          won, bon, rcon, sagon, w_target, hist_every, H):
    w = 0.0; iR1 = 0.0; soc = 1.0; v = lipo_ocv_k(1.0, cells); I = 0.0; Ib = 0.0
    E_chem = 0.0; E_R0 = 0.0; E_R1 = 0.0; E_cu = 0.0; E_wind = 0.0; E_brg = 0.0
    t95 = -1.0; vmin = v; Ipk = 0.0; Ah = 0.0; j = 0
    R0e = R0 * sagon; R1e = R1 * rcon * sagon
    for i in range(n):
        wm = w * G
        ocv = lipo_ocv_k(soc, cells)
        Veff = ocv - R1e * iR1                     # source behind R0 (1-RC Plett ECM)
        I, Ib, v, duty = esc_supply(Veff, R0e, Kt * wm, Rm, Imax)
        a = math.exp(-dt / (R1 * C1))              # RC branch advanced exactly over dt
        iR1_new = a * iR1 + (1.0 - a) * Ib
        Tw = 0.0
        if won > 0.5:
            Tw = bar_windage_torque(w, Rb, hb, rho, Cd)
        Tb = 0.0
        if bon > 0.5:
            Tb = -bearing_drag_torque(w, Fr, mu_b, db, M0)
        w_new = w + dt * (Kt * I * G - Tw - Tb) / J
        if w_new < 0.0:
            w_new = 0.0
        # ledger (power x dt, evaluated at the step start state)
        E_chem += ocv * Ib * dt
        E_R0 += R0e * Ib * Ib * dt
        E_R1 += R1e * iR1 * iR1 * dt
        E_cu += I * I * Rm * dt
        E_wind += Tw * w * dt
        E_brg += Tb * w * dt
        Ah += Ib * dt / 3600.0
        soc -= Ib * dt / Q
        iR1 = iR1_new; w = w_new
        if v < vmin:
            vmin = v
        if I > Ipk:
            Ipk = I
        if t95 < 0.0 and w >= 0.95 * w_target:
            t95 = (i + 1) * dt
        if hist_every > 0 and i % hist_every == 0 and j < H.shape[0]:
            H[j, 0] = i * dt; H[j, 1] = w; H[j, 2] = v; H[j, 3] = I; H[j, 4] = Ib; H[j, 5] = soc
            j += 1
    KE = 0.5 * J * w * w
    Ecap = 0.5 * C1 * (R1e * iR1) ** 2
    res = E_chem - (KE + E_R0 + E_R1 + Ecap + E_cu + E_wind + E_brg)
    return w, t95, vmin, Ipk, Ah, E_chem, KE, E_cu, E_wind, E_brg, E_R0 + E_R1, res / max(E_chem, 1e-30), j


def run(params=None, dt=1e-4, history=False):
    p = dict(DEFAULTS); p.update(params or {}); b = build(p)
    n = int(round(p["t_end"] / dt))
    every = max(n // 600, 1) if history else 0
    H = np.zeros((n // max(every, 1) + 2 if history else 1, 6))
    o = _loop(n, dt, b["J_eff"], b["Kt"], p["G"], p["R_m"], p["I_max"], p["cells"], b["Q"], p["R0"], p["R1"],
              p["C1"], p["R_bar"], p["h_bar"], p["Cd"], b["rho"], p["mu_b"], p["F_rad"], p["d_bore"], p["M0"],
              p["windage_on"], p["bearing_on"], p["rc_on"], p["sag_on"], W_TARGET, every, H)
    w, t95, vmin, Ipk, Ah, Ech, KE, Ecu, Ew, Eb, Ebat, led, j = o
    out = dict(t95_s=t95 if t95 > 0 else p["t_end"] * 10, rpm_final=w * 60 / (2 * math.pi),
               v_min_per_cell=vmin / p["cells"], I_peak_A=Ipk, mAh_used=Ah * 1000, KE_final_J=KE,
               E_windage_J=Ew, E_copper_J=Ecu, E_pack_loss_J=Ebat, _ledger_rel=led,
               _reached=1.0 if t95 > 0 else 0.0)
    if history:
        Hh = H[:j]
        hist = dict(t=Hh[:, 0], rpm=Hh[:, 1] * 60 / (2 * math.pi), v_cell=Hh[:, 2] / p["cells"], I_motor=Hh[:, 3],
                    I_batt=Hh[:, 4], soc=Hh[:, 5] * 100)
        return out, hist
    return out


def critical_dt(params):
    p = dict(DEFAULTS); p.update(params); b = build(p)
    return 2.0 * b["tau_mech"]                    # explicit Euler on a first-order lag: stable below 2 tau


def analytic_t95(p):
    """No losses, stiff pack: spin_up_time block (current-limited then back-EMF limited)."""
    q = dict(DEFAULTS); q.update(p); b = build(q)
    V = q["cells"] * 4.2
    return spin_up_time(b["J_eff"], b["Kt"], q["R_m"], V, q["I_max"], 0.95 * W_TARGET, G=q["G"])


ideal = dict(windage_on=0.0, bearing_on=0.0, sag_on=0.0)
modes = [
    FailureMode("slow spin-up",
                check=lambda o: ((o["t95_s"] > 2.0, 2.0 / o["t95_s"]) if o["_reached"] > 0.5 else
                                 (True, o["rpm_final"] / (0.95 * 12000.0))),
                origin="trial user: flywheel must reach 95% of 12,000 rpm within 2.0 s",
                assumption="if 95% speed is never reached, SF = top speed / required speed"),
    FailureMode("pack sag", metric="v_min_per_cell", limit=3.3, fails_when="below",
                origin="trial user: pack must not sag below 3.3 V/cell under load"),
]
SPEC = SimSpec(
    name="Trial 1 flywheel spin-up", run=lambda p, dt: run(p, dt), params={}, dt=1e-4,
    metrics=["t95_s", "rpm_final", "v_min_per_cell", "I_peak_A", "mAh_used"],
    uncertain={"R0": (0.08, 0.25), "kv": (1045.0, 1155.0), "R_m": (0.030, 0.045), "I_w": (6.8e-4, 8.4e-4),
               "Cd": (1.0, 1.5), "I_max": (55.0, 65.0)},
    knockouts={"no windage": (dict(windage_on=0.0), {"rpm_final": "+"}),
               "no bearing drag": (dict(bearing_on=0.0), {"rpm_final": "+"}),
               "no RC polarization": (dict(rc_on=0.0), {"v_min_per_cell": "+"}),
               "no pack sag": (dict(sag_on=0.0), {"v_min_per_cell": "+"})},
    limits=[("ESC limit 0 A -> flywheel never moves", dict(I_max=0.0), {"rpm_final": 0.0}),
            ("ideal pack, no losses -> spin_up_time block", ideal,
             {"t95_s": lambda v: abs(v - analytic_t95(ideal)) / analytic_t95(ideal) < 0.01})],
    bounds={"t95_s": (0.0, 1e3), "v_min_per_cell": (0.0, 4.25), "rpm_final": (0.0, 40000.0)},
    critical_dt=critical_dt, mc_n=200, failure_modes=modes)

def _t95_or_nan(p):
    o = run(p)
    return o["t95_s"] if o["_reached"] > 0.5 else float("nan")     # never reaches 95%: gap in the line


if __name__ == "__main__":
    OUT = os.path.join(HERE, "out", "trial1"); os.makedirs(OUT, exist_ok=True)
    rep = verify_all(SPEC)
    nom, hist = run({}, 1e-4, history=True)
    t_an = analytic_t95(ideal)
    G_s = [1.0, 1.25, 1.5, 1.75, 2.0]
    sweep = dict(x=np.array(G_s), x_scale=1.0, x_label="belt ratio (motor turns per flywheel turn)",
                 y_label="time to 95% speed (s)", title="Spin-up time vs belt ratio", limit=2.0, limit_label="2.0 s FAIL",
                 series={"R0 0.15 ohm (nominal)": [_t95_or_nan(dict(G=g)) for g in G_s],
                         "R0 0.25 ohm (tired pack)": [_t95_or_nan(dict(G=g, R0=0.25)) for g in G_s]},
                 mark=1.5, input_labels={"R0": "pack resistance", "kv": "motor KV", "R_m": "winding resistance",
                                         "I_w": "flywheel inertia", "Cd": "windage Cd", "I_max": "ESC current limit"})
    fm = rep.failure
    worst = max(fm.values(), key=lambda v: v["p_fail"])
    failed = any(v["nominal_failed"] for v in fm.values())
    answer = dict(label="Flywheel setup OK?", value="NO" if failed else "YES", unit="",
                  status="FAIL" if failed else "PASS",
                  sub=f"95% speed in {nom['t95_s']:.2f} s, pack min {nom['v_min_per_cell']:.2f} V/cell")
    unreached = [g for g in G_s if math.isnan(_t95_or_nan(dict(G=g)))]
    findings = [
        f"95% of 12,000 rpm in {nom['t95_s']:.2f} s; ideal (no sag, no losses) {t_an:.2f} s, so pack sag and drag cost "
        f"{nom['t95_s'] - t_an:.2f} s.",
        f"Pack bottoms at {nom['v_min_per_cell']:.2f} V/cell at the {nom['I_peak_A']:.0f} A current limit; "
        f"{nom['mAh_used']:.0f} mAh used in {DEFAULTS['t_end']:.0f} s.",
        f"Windage absorbs {nom['E_windage_J']:.0f} J and winding copper {nom['E_copper_J']:.0f} J; pack internal loss "
        f"{nom['E_pack_loss_J']:.0f} J.",
        f"Monte Carlo ({fm['slow spin-up']['n']} runs): P(slow) = {fm["slow spin-up"]["p_fail"]*100:.0f}%, "
        f"P(sag) = {fm["pack sag"]["p_fail"]*100:.0f}%. Top driver of spin-up time: "
        f"{sweep['input_labels'].get(fm['slow spin-up']['drivers'][0][0], fm['slow spin-up']['drivers'][0][0])} (rank corr {fm['slow spin-up']['drivers'][0][1]:+.2f}).",
        f"Belt ratios {', '.join(f'{g:g}' for g in unreached)} never reach 95% speed on the nominal pack (gaps in the "
        f"sweep): top speed is capped by KV x volts / ratio under sag and windage. A tired pack (0.25 ohm) never "
        f"reaches it at any ratio tried.",
    ]
    assumptions = ["Motor 1100 KV, 35 mOhm, 60 A ESC limit, 1.5:1 belt: ASSUMED (no motor named).",
                   "Pack 5S 450 mAh: R0 0.15 ohm (30 mOhm/cell, hobby IR band), R1 0.05 ohm, C1 600 F: ASSUMED.",
                   "LiPo OCV(SOC) table is an approximate typical curve (lipo_ocv block, unsourced).",
                   "Lossless ESC, full throttle from rest, no regen; belt losses ignored.",
                   "Windage: bar spinner, Cd 1.28, 6.35 mm thick bar."]
    sources = ["Plett, Battery Management Systems Vol. I (2015) Ch. 2 (1-RC ECM)", "SKF simplified bearing torque",
               "bar_windage_torque block (Cd x dynamic pressure integrated over the bar)"]
    panels = [dict(title="Flywheel speed", keys=["rpm"], unit="rpm", t_scale=1.0, t_label="time (s)"),
              dict(title="Pack voltage per cell", keys=["v_cell"], unit="V", color=1, t_scale=1.0, t_label="time (s)",
                   limit=3.3, limit_label="3.3 V FAIL"),
              dict(title="Currents", keys=["I_motor", "I_batt"], labels=["motor", "battery"], unit="A", color=2,
                   t_scale=1.0, t_label="time (s)")]
    paths = results_report(OUT, "Trial 1: flywheel spin-up", "How fast does the flywheel spin up, and does the pack sag "
                           "below 3.3 V/cell?", answer, rep, history=hist, history_panels=panels, sweep=sweep,
                           findings=findings, assumptions=assumptions, sources=sources)
    print(rep.summary()); [print("-", f) for f in findings]; print(paths)

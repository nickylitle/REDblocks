"""TRIAL 2: does the blade shaft take a permanent bend on a big hit?
Prompt (as a user would give it): "My 600 J horizontal bar rides on bearings on a 10 mm dead shaft of 4140 prehard,
30 mm between the two frame plates. If the tooth (2 mm corner radius) hits an AR500 target square at 12,000 rpm,
does the shaft bend permanently more than 0.1 mm? Compare 7075 and 4140 normalized."
Blocks: circle + moment_curvature_polygon -> hinge_table -> hinge_to_spring_table -> tabular_return (shaft),
simply_supported_center (Roark 8.1), effective_modulus + flat_punch_stiffness + tabular_return (plastic contact,
fully plastic mean pressure 3Y), load_curve (published fit or flagged fallback), spinner_strike (anchor)."""
import math, os, sys
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from simblocks.materials import get, load_curve
from simblocks.structures.sections.properties import circle
from simblocks.structures.sections.moment_curvature import moment_curvature_polygon, hinge_table
from simblocks.structures.beams.static import simply_supported_center
from simblocks.constitutive.plasticity import tabular_return, hinge_to_spring_table
from simblocks.contact.hertz import effective_modulus, flat_punch_stiffness
from simblocks.contact.penalty import spinner_strike
from simblocks.diagnostics.verify import SimSpec, verify_all
from simblocks.diagnostics.failure import FailureMode
from simblocks.viz.report import results_report
from simblocks.viz.diagram import Schematic

DEFAULTS = dict(shaft_mat="4140 prehard", d=0.010, L=0.030, rpm=12000.0, I_w=7.6e-4, R_tip=0.09, m_blade=0.28,
                m_machine=1.361, m_target=1.361, R_tooth=0.002, tooth_mat="S7", target_mat="AR500", t_target=0.004,
                shaft_pl_on=1.0, contact_pl_on=1.0, t_end=4e-4)


def build(p):
    sm = get_mat(p["shaft_mat"])
    curve = load_curve(p["shaft_mat"])
    sec = circle(p["d"])
    ss = simply_supported_center(sm, sec, p["L"])                       # Roark 8.1 case 1a: k = 48EI/L^3
    th = np.linspace(0, 2 * math.pi, 181)[:-1]
    xy = np.c_[0.5 * p["d"] * np.cos(th), 0.5 * p["d"] * np.sin(th)]
    mk = moment_curvature_polygon(curve, xy, kappa_max=0.3 / p["d"])
    k_th, thp_tab, M_tab = hinge_to_spring_table(hinge_table(mk, p["d"]))   # hinge length = d
    # 3-hinge mechanism of a simply supported beam with a centre load: F = 4M/L, centre deflection dp = thp L/4
    s_acc = thp_tab * p["L"] / 4.0
    s_F = M_tab * 4.0 / p["L"]
    if p["shaft_pl_on"] < 0.5:
        s_F = s_F * 0 + 1e12
    # contact: elastic flat-punch stiffness at the tooth radius, then fully plastic indentation of the softer side,
    # F = 3 Y pi a^2 with a^2 = 2 R delta_p (Johnson, Contact Mechanics ch. 6)
    tm, om = get_mat(p["tooth_mat"]), get_mat(p["target_mat"])
    Es = effective_modulus(tm.E, tm.nu, om.E, om.nu)
    kc = flat_punch_stiffness(p["R_tooth"], Es)
    Y = min(tm.sy, om.sy)
    # contact radius cannot exceed the corner radius: once a^2 = 2 R d_p reaches R^2 the tooth acts as a flat punch
    c_acc = np.linspace(0.0, 0.03, 301)
    a2 = np.minimum(2.0 * p["R_tooth"] * c_acc, p["R_tooth"] ** 2)
    c_F = 3.0 * Y * math.pi * np.maximum(a2, (0.1 * p["R_tooth"]) ** 2)
    if p["contact_pl_on"] < 0.5:
        c_F = c_F * 0 + 1e12
    return dict(k_s=ss["k"], s_acc=s_acc, s_F=np.ascontiguousarray(s_F), k_c=kc, c_acc=c_acc,
                c_F=np.ascontiguousarray(c_F), Y=Y, Mp=float(M_tab[0]), curve_source=curve.source, sm=sm, Mp_full=ss["Mp"])


def get_mat(text):
    from simblocks.materials.curves import resolve_material
    return get(resolve_material(text))


@njit(cache=True)
def _loop(n, dt, mb, Iw, R, ma, mo, w0, ks, s_acc, s_F, kc, c_acc, c_F, t_target, every, H):
    perf = 0.0
    xb = 0.0; vb = 0.0; phi = 0.0; w = w0; xa = 0.0; va = 0.0; xo = 0.0; vo = 0.0
    sp = 0.0; sacc = 0.0; cp = 0.0; cacc = 0.0
    Wsp = 0.0; Wcp = 0.0; Fcmax = 0.0; Fsmax = 0.0; j = 0
    for i in range(n):
        pen = xb + R * phi - xo
        sp_old = sp; cp_old = cp
        if perf < 0.5:
            Fc, cp, cacc = tabular_return(pen, cp, cacc, kc, c_acc, c_F, 1)
            if cp >= t_target:                      # tooth indented through the armor: punched out, contact ends
                perf = 1.0
                Wcp += Fc * (cp - cp_old) + Fc * Fc / (2 * kc)
                cp_old = cp
                Fc = 0.0
        else:
            Fc = 0.0
        Fs, sp, sacc = tabular_return(xb - xa, sp, sacc, ks, s_acc, s_F, 0)
        Wcp += Fc * (cp - cp_old)
        Wsp += Fs * (sp - sp_old)
        vb += dt * (-Fc - Fs) / mb
        w += dt * (-Fc * R) / Iw
        va += dt * Fs / ma
        vo += dt * Fc / mo
        xb += vb * dt; phi += w * dt; xa += va * dt; xo += vo * dt
        if Fc > Fcmax:
            Fcmax = Fc
        if abs(Fs) > Fsmax:
            Fsmax = abs(Fs)
        if every > 0 and i % every == 0 and j < H.shape[0]:
            H[j, 0] = i * dt; H[j, 1] = Fc; H[j, 2] = Fs; H[j, 3] = sp; H[j, 4] = xb - xa; H[j, 5] = w
            j += 1
    Fc = 0.0
    if perf < 0.5:
        Fc, _, _ = tabular_return(xb + R * phi - xo, cp, cacc, kc, c_acc, c_F, 1)
    Fs, _, _ = tabular_return(xb - xa, sp, sacc, ks, s_acc, s_F, 0)
    KE = 0.5 * mb * vb * vb + 0.5 * Iw * w * w + 0.5 * ma * va * va + 0.5 * mo * vo * vo
    E0 = 0.5 * Iw * w0 * w0
    Eel = Fc * Fc / (2 * kc) + Fs * Fs / (2 * ks)
    return sp, cp, Fcmax, Fsmax, 0.5 * mo * vo * vo, w, (E0 - KE - Eel - Wsp - Wcp) / E0, Wsp, Wcp, j, perf


def run(params=None, dt=2e-8, history=False):
    p = dict(DEFAULTS); p.update(params or {}); b = build(p)
    n = int(round(p["t_end"] / dt)); every = max(n // 800, 1) if history else 0
    H = np.zeros((n // max(every, 1) + 2 if history else 1, 6))
    w0 = p["rpm"] * 2 * math.pi / 60
    ma = p["m_machine"] - p["m_blade"]
    sp, cp, Fc, Fs, Eo, w, led, Wsp, Wcp, j, perf = _loop(n, dt, p["m_blade"], p["I_w"], p["R_tip"], ma, p["m_target"], w0,
                                                          b["k_s"], b["s_acc"], b["s_F"], b["k_c"], b["c_acc"],
                                                          b["c_F"], p["t_target"], every, H)
    out = dict(shaft_set_mm=abs(sp) * 1e3, dent_mm=cp * 1e3, F_contact_peak_kN=Fc / 1e3, F_shaft_peak_kN=Fs / 1e3,
               E_to_target_J=Eo, rpm_after=w * 60 / (2 * math.pi), W_shaft_plastic_J=Wsp, W_dent_J=Wcp,
               perforated=perf, _ledger_rel=led)
    if history:
        Hh = H[:j]
        return out, dict(t=Hh[:, 0], Fc=Hh[:, 1] / 1e3, Fs=Hh[:, 2] / 1e3, set=Hh[:, 3] * 1e3, defl=Hh[:, 4] * 1e3,
                         rpm=Hh[:, 5] * 60 / (2 * math.pi))
    return out


def critical_dt(params):
    p = dict(DEFAULTS); p.update(params); b = build(p)
    mstar = 1 / (1 / (p["I_w"] / p["R_tip"] ** 2) + 1 / p["m_target"])
    return min(2 / math.sqrt(b["k_c"] / mstar), 2 / math.sqrt(b["k_s"] / p["m_blade"]))


def rigid_anchor():
    """spinner_strike block: fully plastic (e = 0) rigid-body impulse, Frame free. Energy absorbed
    should match the sim's plastic work + residual vibration once the shaft is made rigid."""
    p = DEFAULTS
    s = spinner_strike(p["I_w"], p["rpm"] * 2 * math.pi / 60, p["R_tip"], p["m_target"], e=0.0)
    return s


modes = [FailureMode("shaft bend", metric="shaft_set_mm", limit=0.1, fails_when="above",
                     origin="trial user: shaft fails if it takes a permanent bend over 0.1 mm at the blade")]
SPEC = SimSpec(
    name="Trial 2 blade shaft bend", run=lambda p, dt: run(p, dt), params={}, dt=2e-8,
    metrics=["shaft_set_mm", "F_contact_peak_kN", "F_shaft_peak_kN", "E_to_target_J"],
    uncertain={"rpm": (10800.0, 13200.0), "m_target": (1.1, 1.5), "R_tooth": (0.001, 0.003), "L": (0.025, 0.035),
               "t_target": (0.003, 0.006)},
    knockouts={"no shaft plasticity": (dict(shaft_pl_on=0.0), {"shaft_set_mm": "-"}),
               "elastic contact": (dict(contact_pl_on=0.0), {"F_contact_peak_kN": "+"})},
    limits=[("blade not spinning -> nothing happens", dict(rpm=1e-6), {"shaft_set_mm": 0.0})],
    bounds={"shaft_set_mm": (0.0, 15.0), "E_to_target_J": (0.0, 600.0), "dent_mm": (0.0, 30.0)},
    critical_dt=critical_dt, mc_n=150, failure_modes=modes)

if __name__ == "__main__":
    OUT = os.path.join(HERE, "out", "trial2"); os.makedirs(OUT, exist_ok=True)
    rep = verify_all(SPEC)
    nom, hist = run({}, 2e-8, history=True)
    b = build(dict(DEFAULTS))
    ds = [0.008, 0.010, 0.012, 0.014]
    mats = {"4140 prehard (890 MPa yield)": "4140 prehard", "4140 normalized (655 MPa)": "4140_normalized",
            "7075-T6 (503 MPa)": "7075"}
    series = {lab: [run(dict(d=d, shaft_mat=m), 4e-8)["shaft_set_mm"] for d in ds] for lab, m in mats.items()}
    sweep = dict(x=np.array(ds), x_scale=1e3, x_label="shaft diameter (mm)", y_label="permanent shaft bend (mm)",
                 title="Permanent bend vs shaft diameter and material", limit=0.1, limit_label="0.1 mm FAIL",
                 series=series, mark=0.010,
                 input_labels={"rpm": "blade rpm", "m_target": "target mass", "R_tooth": "tooth corner radius",
                               "L": "shaft span", "t_target": "target armor thickness"})
    fm = rep.failure["shaft bend"]
    answer = dict(label="Shaft bends > 0.1 mm?", value="YES" if fm["nominal_failed"] else "NO", unit="",
                  status="FAIL" if fm["nominal_failed"] else "PASS",
                  sub=f"permanent bend {nom['shaft_set_mm']:.3f} mm, peak shaft load {nom['F_shaft_peak_kN']:.1f} kN")
    an = rigid_anchor()
    d_ok = [f"{d*1e3:.0f}" for d, v in zip(ds, series["4140 prehard (890 MPa yield)"]) if v < 0.1]
    findings = [
        f"Square hit at {DEFAULTS['rpm']:.0f} rpm: peak tooth force {nom['F_contact_peak_kN']:.1f} kN, peak shaft "
        f"load {nom['F_shaft_peak_kN']:.1f} kN, permanent bend {nom['shaft_set_mm']:.3f} mm.",
        f"Shaft plastic hinge: Mp = {b['Mp_full']:.0f} N m on the 10 mm {b['sm'].name} shaft "
        f"(collapse load 4 Mp / L = {4*b['Mp_full']/DEFAULTS['L']/1e3:.1f} kN; first yield at {b['Mp']:.0f} N m).",
        f"Nominal hit {'PUNCHES THROUGH' if nom['perforated'] else 'does not punch through'} the {DEFAULTS['t_target']*1e3:.0f} mm "
        f"AR500 (indentation {nom['dent_mm']:.1f} mm).",
        f"{nom['E_to_target_J']:.0f} J goes into the target, {nom['W_dent_J']:.0f} J into the contact dent, "
        f"{nom['W_shaft_plastic_J']:.1f} J into bending the shaft. Upper bound from the rigid-body anchor (spinner_strike, e = 0, no punch-through): "
        f"{an['E_absorbed']:.0f} J absorbed in the collision.",
        f"Monte Carlo ({fm['n']} hits): P(bend > 0.1 mm) = {fm['p_fail']*100:.0f}%; top driver "
        f"{sweep['input_labels'].get(fm['drivers'][0][0], fm['drivers'][0][0])} ({fm['drivers'][0][1]:+.2f}).",
        "4140 prehard diameters that stay under 0.1 mm: " + (", ".join(d_ok) + " mm" if d_ok else "none tried") +
        ". The corrected 4140 prehard yield (890 vs the old 655 MPa) is what separates it from 4140 normalized.",
    ]
    assumptions = ["Square hit: contact force perpendicular to the shaft at the blade's mid-span; blade on bearings.",
                   "Shaft simply supported between frame plates 30 mm apart (real plates add some end fixity).",
                   "Plastic hinge length = shaft diameter; 4140 prehard curve is the flagged Ramberg-Osgood fallback "
                   "(no published fit for this temper).",
                   "Tooth contact: flat-punch stiffness at the 2 mm corner radius, then fully plastic indentation of the "
                   "softer side (3Y, Tabor / Johnson ch. 6).",
                   "Target free (1.36 kg), Frame free; no friction, no rotation of either chassis.",
                   "Target armor 4 mm AR500: once the tooth has indented through the thickness the plate is "
                   "treated as punched out and contact ends (no petalling or plug-shear model yet)."]
    sources = ["Roark 7e Table 8.1 (simply supported, centre load)", "Johnson, Contact Mechanics ch. 3 and 6",
               "moment_curvature from the material curve (fibre integration)", "spinner_strike impulse block"]
    sch = Schematic("Trial 2: blade shaft strike, physical model",
                    "Bar spinner on a dead shaft hits a free target; values from build()")
    sch.body("blade", "rotor", "Blade on bearings", info=[f"I = {DEFAULTS['I_w']:.2e} kg m^2", f"m = {DEFAULTS['m_blade']} kg"])
    sch.body("target", "mass", "Target", info=[f"m = {DEFAULTS['m_target']} kg"])
    sch.body("frame", "mass", "Frame", info=[f"m = {DEFAULTS['m_machine'] - DEFAULTS['m_blade']:.2f} kg"])
    sch.link("blade", "target", "contact", "Tooth contact", f"k = {b['k_c']:.2e} N/m\nplastic 3Y = {3*b['Y']/1e9:.2f} GPa",
             "flat_punch_stiffness + tabular_return")
    sch.link("blade", "frame", "elastoplastic", "Shaft bending", f"k = {b['k_s']:.2e} N/m\nMp = {b['Mp_full']:.0f} N m (yield {b['Mp']:.0f})",
             "simply_supported_center + moment_curvature_polygon")
    sch.probe("frame", "permanent shaft bend")
    sch_paths = sch.draw(os.path.join(OUT, "schematic"))
    panels = [dict(title="Tooth contact force", keys=["Fc"], unit="kN", t_scale=1e3, t_label="time (ms)"),
              dict(title="Shaft force", keys=["Fs"], unit="kN", color=1, t_scale=1e3, t_label="time (ms)"),
              dict(title="Shaft deflection and permanent set", keys=["defl", "set"], labels=["total", "permanent"],
                   unit="mm", color=2, t_scale=1e3, t_label="time (ms)")]
    paths = results_report(OUT, "Trial 2: blade shaft bend", "Does a 10 mm 4140 prehard dead shaft take a permanent "
                           "bend over 0.1 mm on a square 600 J strike?", answer, rep, history=hist, history_panels=panels,
                           sweep=sweep, findings=findings, assumptions=assumptions, sources=sources,
                           extra_svgs=[sch_paths[1]])
    print(rep.summary()); [print("-", f) for f in findings]; print("material law:", b["curve_source"][:120])

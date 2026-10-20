"""Builds apollo11.html: runs the two Apollo 11 canvases and draws them against NASA's records.

    python build_page.py              (from this folder, or give the path)

Everything on the page is computed here, at build time, from the canvases (apollo11_1_ascent.yaml,
apollo11_2_landing.yaml) and NASA's numbers (nasa_reference.json). The page is one self-contained file: no
internet needed to open it.
"""
import base64
import html
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from simblocks.compiler.system import load_system          # noqa: E402

FT = 0.3048
NMI = 1852.0
LB = 0.45359237
REF = json.load(open(os.path.join(HERE, "nasa_reference.json")))

# ---------------------------------------------------------------------------------------------- the runs
print("running the ascent ...")
ASC = load_system(os.path.join(HERE, "apollo11_1_ascent.yaml"))
RA, HA = ASC.run(history=True, n_log=3000)
print("running the landing, as flown ...")
LAND = load_system(os.path.join(HERE, "apollo11_2_landing.yaml"))
RL, HL = LAND.run(history=True, n_log=3000)
print("running the landing, automatic ...")
RN, HN = LAND.run({"crew": 0}, history=True, n_log=3000)
CA = ASC.nominal.ctx
CL = LAND.nominal.ctx
T_FM = float(CA["t_fm"])

# ascent series (range time: the model's clock starts at first motion, range time 0.3 s)
ta = HA["t"] + T_FM
rockets = [e for e, i in ASC.nominal.el_info.items() if i["type"] == "rocket"]
thrust = sum(HA[f"{e}.thrust"] for e in rockets)
drag = HA["air.drag"]
mass_a = HA["s4.vehicle_mass"]
asc = dict(t=ta, alt=HA["guidance.altitude"] / 1e3, v=HA["guidance.speed_inertial"], vE=HA["guidance.airspeed"],
           fpa=HA["guidance.flight_path_angle"], range=HA["guidance.downrange"] / 1e3,
           q=HA["air.dynamic_pressure"] / 1e3, mass=mass_a / 1e3, acc=(thrust - drag) / mass_a, mach=HA["air.mach"])


def landing_series(H):
    return dict(t=H["t"], alt=H["p63.altitude"] / FT, v=H["lm.speed"] / FT, vu=H["p63.speed_up"] / FT,
                vf=H["p63.speed_forward"] / FT, range=H["p63.range_to_site"] / NMI, thr=H["dps_engine.throttle"] * 100,
                prop=H["dps_engine.propellant"] / LB, mass=H["dps_engine.vehicle_mass"])


lf = landing_series(HL)
la = landing_series(HN)
t_td, t_td_auto = RL["t_touchdown"], RN["t_touchdown"]


def at(s, key, t):
    return float(np.interp(t, s["t"], s[key]))


def phase_times(H):
    out = {}
    for name in ("p63", "p64", "p66_crew", "p66_auto"):
        on = H[f"{name}.throttle"] > 0
        if on.any():
            out[name] = (float(H["t"][np.argmax(on)]), float(H["t"][len(on) - 1 - np.argmax(on[::-1])]))
    return out


PH_F = phase_times(HL)
PH_A = phase_times(HN)

# ---------------------------------------------------------------------------------------------- formatting
def f(x, d=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:,.{d}f}"


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def pct(sim, ref):
    return 100.0 * (sim - ref) / ref


def esc(s):
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------------------------------- charts (SVG)
COL = dict(sim="#f2f5fa", auto="#7f8ea3", nasa="#ff8a3d", flight="#5ad1ff", call="#ffd166", grid="#1a2230",
           axis="#3a4659", ink="#9aa7b8", ref="#46536a")


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(span / s - n))
    t0 = math.ceil(lo / step - 1e-9) * step
    out = []
    x = t0
    while x <= hi + 1e-9 * step:
        out.append(round(x, 10))
        x += step
    return out


def tick_label(v, step):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    d = 0 if step >= 1 else (1 if step >= 0.1 else 2)
    return f"{v:.{d}f}"


CHART_DATA = []
YLAB = {}


def chart(cid, lines, x_dom, y_dom, xlab, ylab, points=(), vlines=(), W=560, H=300, xticks=None, yticks=None,
          hover=True, xfmt="{:.0f}", yunit="", xunit="s", bands=(), W_=None):
    """lines: (name, x, y, color, width, dash). points: (name, x, y, color, shape, label). vlines: (x, label)."""
    yt = yticks or nice_ticks(*y_dom, 5)
    xt = xticks or nice_ticks(*x_dom, 6)
    ystep = (yt[1] - yt[0]) if len(yt) > 1 else 1
    xstep = (xt[1] - xt[0]) if len(xt) > 1 else 1
    ylabs = [tick_label(v, ystep) for v in yt]
    mL = 12 + 7.0 * max(len(s) for s in ylabs) + 6
    mR, mT, mB = 14, 30 if vlines else 14, 40
    X = lambda x: mL + (x - x_dom[0]) / (x_dom[1] - x_dom[0]) * (W - mL - mR)
    Y = lambda y: mT + (y_dom[1] - y) / (y_dom[1] - y_dom[0]) * (H - mT - mB)
    o = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(ylab)} against {esc(xlab)}">']
    for (a, b, label) in bands:
        o.append(f'<rect x="{X(a):.1f}" y="{mT}" width="{X(b) - X(a):.1f}" height="{H - mT - mB}" fill="#0f1621"/>')
        o.append(f'<text x="{(X(a) + X(b)) / 2:.1f}" y="{H - mB - 8}" class="band">{esc(label)}</text>')
    for v, s in zip(yt, ylabs):
        o.append(f'<line x1="{mL}" x2="{W - mR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{COL["grid"]}"/>')
        o.append(f'<text x="{mL - 8}" y="{Y(v):.1f}" class="tk" text-anchor="end" dy="0.33em">{s}</text>')
    for v in xt:
        o.append(f'<text x="{X(v):.1f}" y="{H - mB + 18}" class="tk" text-anchor="middle">{tick_label(v, xstep)}</text>')
    o.append(f'<line x1="{mL}" x2="{W - mR}" y1="{H - mB}" y2="{H - mB}" stroke="{COL["axis"]}"/>')
    o.append(f'<text x="{W - mR}" y="{H - 4}" class="ax" text-anchor="end">{esc(xlab)}</text>')
    # event lines, labels moved down a row when they would meet
    rows = []
    for x, label in vlines:
        xx = X(x)
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{mT}" y2="{H - mB}" stroke="{COL["ref"]}" stroke-dasharray="3 4"/>')
        w = 6.4 * len(label)
        left = min(max(xx - w / 2, mL), W - mR - w)
        r = 0
        while any(rr == r and not (left > l2 + w2 + 6 or left + w + 6 < l2) for (l2, w2, rr) in rows):
            r += 1
        rows.append((left, w, r))
        o.append(f'<text x="{left:.1f}" y="{mT - 10 + 13 * r}" class="ev">{esc(label)}</text>')
    o.append(f'<g clip-path="url(#clip-{cid})"><clipPath id="clip-{cid}"><rect x="{mL}" y="{mT - 2}" width="{W - mL - mR}" height="{H - mT - mB + 4}"/></clipPath>')
    YLAB[cid] = ylab
    data = dict(id=cid, mL=mL, mR=mR, mT=mT, mB=mB, W=W, H=H, x0=x_dom[0], x1=x_dom[1], y0=y_dom[0], y1=y_dom[1],
                xunit=xunit, yunit=yunit, xfmt=xfmt, lines=[], points=[])
    for (name, x, y, color, width, dash) in lines:
        x = np.asarray(x); y = np.asarray(y)
        keep = (x >= x_dom[0] - 1e-9) & (x <= x_dom[1] + 1e-9)
        x, y = x[keep], y[keep]
        if len(x) > 900:
            idx = np.unique(np.linspace(0, len(x) - 1, 900).astype(int)); x, y = x[idx], y[idx]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(zip(x, y)))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{da}/>')
        data["lines"].append(dict(name=name, color=color, x=[round(float(a), 3) for a in x], y=[round(float(b), 4) for b in y]))
    o.append("</g>")
    for (name, x, y, color, shape, label) in points:
        if shape == "diamond":
            s = 5
            o.append(f'<path d="M{X(x):.1f},{Y(y) - s:.1f} L{X(x) + s:.1f},{Y(y):.1f} L{X(x):.1f},{Y(y) + s:.1f} L{X(x) - s:.1f},{Y(y):.1f} Z" fill="{color}" stroke="#05070b" stroke-width="1.5"/>')
        elif shape == "ring":
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="none" stroke="{color}" stroke-width="2.2"/>')
        else:
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{color}" stroke="#05070b" stroke-width="1.5"/>')
        data["points"].append(dict(name=name, color=color, x=float(x), y=float(y), label=label))
    if hover:
        o.append(f'<rect class="hit" x="{mL}" y="{mT}" width="{W - mL - mR}" height="{H - mT - mB}" fill="transparent"/>')
    o.append("</svg>")
    CHART_DATA.append(data)
    return "".join(o)


def legend(items):
    out = []
    for name, color, kind in items:
        mark = {"line": f'<i class="lg-line" style="background:{color}"></i>',
                "dash": f'<i class="lg-dash" style="border-color:{color}"></i>',
                "dot": f'<i class="lg-dot" style="background:{color}"></i>',
                "diamond": f'<i class="lg-dia" style="background:{color}"></i>',
                "ring": f'<i class="lg-ring" style="border-color:{color}"></i>',
                "ev": '<i class="lg-ev"></i>'}[kind]
        out.append(f"<span>{mark}{esc(name)}</span>")
    return '<div class="legend">' + "".join(out) + "</div>"


def card(title, note, svg, leg="", wide=False, cid=""):
    import re
    yl = YLAB.get(re.search(r'id="([^"]+)"', svg).group(1), "")
    return (f'<figure class="card{" wide" if wide else ""}"{f" id={chr(34)}{cid}{chr(34)}" if cid else ""}>'
            f'<figcaption><h3>{esc(title)}</h3><p>{note}</p></figcaption>{leg}<div class="yl">{esc(yl)}</div>'
            f'<div class="plot">{svg}<div class="tip" hidden></div></div></figure>')


# ---------------------------------------------------------------------------------------------- ascent charts
A = REF["ascent"]
ev_a = [(t, n) for n, t in A["events"]]
cp = {row[5]: row for row in A["checkpoints"]}
nasa_pts = lambda key: [("NASA", row[3], row[4], COL["nasa"], "dot", row[0]) for row in A["checkpoints"] if row[2] == key]
asc_charts = []
asc_charts.append(card("Altitude", "Kilometres above the pad radius. The sim flies the whole climb from engine chemistry, masses and gravity.",
                       chart("a-alt", [("REDblocks", asc["t"], asc["alt"], COL["sim"], 2.2, "")], (0, 720), (0, 210), "range time, s",
                             "altitude, km", nasa_pts("alt"), ev_a, yunit="km"), cid="a-alt-card"))
asc_charts.append(card("Space-fixed speed", "Metres per second against the stars. It starts at 409 m/s: the speed Kennedy Space Center already has from Earth's spin."
                       .replace("409", f"{asc['v'][0]:.0f}"),
                       chart("a-v", [("REDblocks", asc["t"], asc["v"], COL["sim"], 2.2, "")], (0, 720), (0, 8000), "range time, s",
                             "speed, m/s", nasa_pts("v"), ev_a, yunit="m/s")))
asc_charts.append(card("Acceleration", "Thrust minus drag, over mass. Each peak is the moment before an engine cutoff, when the stage is nearly empty.",
                       chart("a-acc", [("REDblocks", asc["t"], asc["acc"], COL["sim"], 2.2, "")], (0, 720), (0, 42), "range time, s",
                             "acceleration, m/s²", [("NASA peak", t, a, COL["nasa"], "dot", f"{n} peak") for n, t, a in A["peaks"]],
                             ev_a, yunit="m/s²")))
asc_charts.append(card("Dynamic pressure", "Aerodynamic load in kilopascals. The one clear miss: the sim uses the standard atmosphere and a generic drag curve.",
                       chart("a-q", [("REDblocks", asc["t"], asc["q"], COL["sim"], 2.2, "")], (0, 200), (0, 40), "range time, s",
                             "dynamic pressure, kPa", nasa_pts("q"), [(162.3, "S-IC off")], yunit="kPa")))
asc_charts.append(card("Flight path angle", "Degrees above the local horizon. Climbing steeply, then flattening into orbit.",
                       chart("a-fpa", [("REDblocks", asc["t"], asc["fpa"], COL["sim"], 2.2, "")], (0, 720), (0, 40), "range time, s",
                             "flight path angle, deg", nasa_pts("fpa"), ev_a, yunit="°")))
asc_charts.append(card("Vehicle mass", "Tonnes. 2,899 t at liftoff, 135 t in orbit. The steps are stages and hardware falling away.",
                       chart("a-m", [("REDblocks", asc["t"], asc["mass"], COL["sim"], 2.2, "")], (0, 720), (0, 3000), "range time, s",
                             "mass, t", nasa_pts("mass"), ev_a, yunit="t")))
asc_track = card("The climb to orbit", "Altitude against ground range, to scale in kilometres. Dots are NASA's measured positions.",
                 chart("a-track", [("REDblocks", asc["range"], asc["alt"], COL["sim"], 2.4, "")], (0, 2700), (0, 210),
                       "ground range, km", "altitude, km",
                       [("NASA", cp[k][4], cp[h][4], COL["nasa"], "dot", cp[h][0].split(":")[0])
                        for h, k in (("h_sep1", "x_sep1"), ("h_sep2", "x_sep2"))],
                       W=1100, H=300, xunit="km", yunit="km"), wide=True)

# ---------------------------------------------------------------------------------------------- landing charts
L = REF["landing"]
plan = L["plan"]["rows"]
calls = L["callouts"]["rows"]
flight = L["flight"]
ev_l = [(PH_F["p64"][0], "P64"), (t_td, "touchdown")]
ev_plan = [(384, "throttle recovery"), (506, "high gate")]
landing_bands = [(0, PH_F["p64"][0], "P63 braking"), (PH_F["p64"][0], PH_F["p66_crew"][0], "P64"),
                 (PH_F["p66_crew"][0], t_td, "P66")]
lines_l = lambda key: [("REDblocks, as flown", lf["t"], lf[key], COL["sim"], 2.2, ""),
                       ("REDblocks, automatic", la["t"], la[key], COL["auto"], 1.6, "5 4")]
plan_pts = lambda col: [("NASA plan", r[1], r[col], COL["nasa"], "dot", r[0]) for r in plan if r[col] is not None]
land_track = card("The descent, side on", "Altitude against range to the landing site. From 48,814 ft and 260 nmi out, to the surface. Eagle flew the last minute by hand.",
                  chart("l-track", [("REDblocks, as flown", lf["range"], lf["alt"], COL["sim"], 2.4, ""),
                                    ("REDblocks, automatic", la["range"], la["alt"], COL["auto"], 1.6, "5 4")],
                        (-262, 2), (0, 52000), "range to the landing site, nmi", "altitude, ft", W=1100, H=320,
                        xunit="nmi", yunit="ft"), wide=True)
land_charts = []
land_charts.append(card("Speed", "Inertial speed, ft/s. The braking phase takes Eagle from orbital speed to a crawl. Dots are NASA's premission trajectory.",
                        chart("l-v", lines_l("v"), (0, 760), (0, 6000), "seconds after ignition", "speed, ft/s",
                              plan_pts(2), [(386, "flight: throttle down"), (506.8, "flight: P64")], yunit="ft/s")))
land_charts.append(card("Altitude", "Feet above the site. Dots: NASA's plan. Rings: the flight's high gate and touchdown.",
                        chart("l-alt", lines_l("alt"), (0, 760), (0, 52000), "seconds after ignition", "altitude, ft",
                              plan_pts(4) + [("flight", 506.8, 7129, COL["flight"], "ring", "P64, high gate (flight)"),
                                             ("flight", 754.7, 0, COL["flight"], "ring", "touchdown (flight)")],
                              ev_plan, yunit="ft")))
land_charts.append(card("Throttle", "Percent of rated thrust. 10 % for 26 s, full throttle, then throttle recovery when the guidance asks for less than 57 %.",
                        chart("l-thr", lines_l("thr"), (0, 760), (0, 100), "seconds after ignition", "throttle, %",
                              [("flight", 386, 57, COL["flight"], "ring", "throttle recovery (flight, 386 s)")],
                              [(26, "full throttle"), (384, "plan: 384 s")], yunit="%")))
land_charts.append(card("Descent rate", "Altitude rate, ft/s (negative is down). Dots: NASA's plan.",
                        chart("l-vu", lines_l("vu"), (0, 760), (-200, 20), "seconds after ignition", "altitude rate, ft/s",
                              plan_pts(3), ev_plan, yunit="ft/s")))
# the last ten minutes of the approach and landing, with the crew's callouts
land_final = card("The last three minutes", "Below 800 ft. Diamonds are the altitudes Aldrin called out, at the times he called them. The sim reaches 400 ft " + f"{620.8 - PH_F['p66_crew'][0]:.0f}" + " s before Eagle did, then flies the crew's descent rates from there.",
                  chart("l-final", lines_l("alt"), (580, 760), (0, 800), "seconds after ignition", "altitude, ft",
                        [("callout", t, h, COL["call"], "diamond", f"called: {h:,} ft") for t, h in calls if h <= 800 and h > 0] +
                        [("flight", 754.7, 0, COL["flight"], "ring", "touchdown (flight)")],
                        [(PH_F["p66_crew"][0], "sim: 400 ft"), (620.8, "flight: 400 ft"), (t_td, "sim: touchdown"),
                         (754.7, "flight: touchdown")],
                        W=1100, H=320, yunit="ft"), wide=True)
land_prop = card("Propellant left", "Pounds in the descent stage tanks. Eagle landed with about 770 lb; the sim's propellant use is 3.4 % lower.",
                 chart("l-prop", lines_l("prop"), (0, 760), (0, 18500), "seconds after ignition", "propellant, lb",
                       [("flight", 754.7, L["propellant_left_lb"], COL["flight"], "ring", "left at touchdown (flight, about 770 lb)")],
                       [], yunit="lb"))
rated = float(CL["dps_rated"])
cmd_f = sum(HL[f"{p}.thrust_command"] for p in ("p63", "p64", "p66_crew", "p66_auto")) / rated * 100
land_cmd = card("Thrust the guidance asks for", "Percent of rated thrust. Early on the law asks for far more than the engine has, so it sits at full throttle; Bennett's report gives about 160 % at the start. When the ask falls under 57 %, the throttle follows it.",
                chart("l-cmd", [("asked for", HL["t"], cmd_f, COL["call"], 1.8, ""), ("throttle", lf["t"], lf["thr"], COL["sim"], 2.2, "")],
                      (0, 760), (0, 180), "seconds after ignition", "percent of rated thrust",
                      [("plan", 0, 160, COL["nasa"], "dot", "about 160 % at ignition [TM]")], [(26, "full throttle"), (386, "flight: throttle down")], yunit="%"))
# ---------------------------------------------------------------------------------------------- numbers
used_flight = L["dps_loaded_lb"] - float(CL["m_doi"]) / LB - L["propellant_left_lb"]
used_sim = RL["prop_used"]
braking = [(r[1], r[2], at(lf, "v", r[1])) for r in plan[1:7]]
worst_v = max(abs(pct(s, n)) for _, n, s in braking)
asc_rows = []
for row in A["checkpoints"]:
    name, unit, _, t, nasa, key, src = row
    sim = RA[key]
    if unit == "t" and key == "m_sep2":
        sim = sim / 1000
    elif unit == "t":
        sim = sim / 1000
    d = sim - nasa
    rel = None if unit in ("deg", "s") else pct(sim, nasa)
    asc_rows.append((name, unit, nasa, sim, d, rel, src))
land_rows = []
for r in plan[:-1]:
    name, t, v, hd, h, dv = r
    vs = at(la, "v", t)
    land_rows.append(("plan", name, t, v, vs, h, at(la, "alt", t) if h is not None else None, "TM table I"))
key_rows = [
    ("Throttle recovery", "s after ignition", 386, RL["t_throttle"], "MR 5.3 (plan 384 s)"),
    ("P64, high gate", "s after ignition", 506.8, RL["t_hg"], "MR 5.3, AL"),
    ("Altitude at high gate", "ft", 7129, at(lf, "alt", RL["t_hg"]), "MR 5.3"),
    ("Descent rate at high gate", "ft/s", -125, at(lf, "vu", RL["t_hg"] - 0.2), "MR 5.3"),
    ("Touchdown", "s after ignition", 754.7, t_td, "MR table 3-1"),
    ("Crew landing longer than automatic", "s", 57, t_td - t_td_auto, "TM: 52 to 62 s"),
    ("Automatic touchdown (the plan)", "s after ignition", 714, t_td_auto, "TM table I"),
    ("Descent propellant used", "lb", used_flight, used_sim, "TM table II, AL"),
    ("Delta-V of the automatic descent", "ft/s", 6775, RN["dv_descent"], "TM table I (plan)"),
]


def table(head, rows, cls=""):
    th = "".join(f'<th class="{c}">{esc(h)}</th>' for h, c in head)
    body = "".join("<tr>" + "".join(f'<td class="{c}">{v}</td>' for (h, c), v in zip(head, r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def errcell(rel, d=None, unit=""):
    if rel is None:
        return f'<span class="mut">{"+" if d >= 0 else "−"}{f(abs(d), 3 if unit == "deg" else 2)} {esc(unit)}</span>'
    c = "ok" if abs(rel) <= 1 else ("warn" if abs(rel) <= 3 else "bad")
    return f'<span class="{c}">{"+" if rel >= 0 else "−"}{abs(rel):.2f} %</span>'


def dp(unit, v):
    return 3 if unit == "deg" else (2 if unit in ("s", "t") or abs(v) < 100 else 1)


asc_table = table([("Checkpoint", ""), ("NASA", "num"), ("REDblocks", "num"), ("Error", "num"), ("Source", "src")],
                  [(esc(n) + f' <span class="mut">{esc(u)}</span>', f(nv, dp(u, nv)), f(sv, dp(u, nv)), errcell(rel, d, u), esc(s))
                   for n, u, nv, sv, d, rel, s in asc_rows])
plan_table = table([("NASA plan event", ""), ("Time, s", "num"), ("Speed, plan", "num"), ("Speed, sim", "num"), ("Error", "num"),
                    ("Altitude, plan", "num"), ("Altitude, sim", "num")],
                   [(esc(n), f(t, 0), f(v, 0), f(vs, 0), errcell(pct(vs, v)) if vs and v > 100 else "", f(h, 0), f(hs, 0))
                    for _, n, t, v, vs, h, hs, _s in land_rows])
key_table = table([("Flight event", ""), ("Unit", "mut"), ("Apollo 11", "num"), ("REDblocks", "num"), ("Difference", "num"), ("Source", "src")],
                  [(esc(n), esc(u), f(a, 1), f(b, 1), f'{"+" if b - a >= 0 else "−"}{f(abs(b - a), 1)}', esc(s)) for n, u, a, b, s in key_rows])

# what the canvases are given, and what they work out
inputs = [
    ("Saturn V liftoff mass, stage and hardware masses", "Flight data", "FE table 20-9"),
    ("F-1 sea-level Isp 263 s, chamber 7.0 MPa, area ratio 16", "Engine rating", "FE section 5"),
    ("J-2 thrust and flow on each stage, propellant used", "Flight data", "FE sections 6, 7"),
    ("Engine cutoff, staging and mixture-shift times", "Flight data", "FE table 2-2"),
    ("Pitch-over rate after the kick: 0.1323 deg/s", "Tuned (the only one)", "set so S-IC staging matches"),
    ("Saturn V drag: 79.5 m², generic Cd against Mach", "Assumption", "not wind-tunnel data"),
    ("Earth: point-mass gravity, spin, US Standard Atmosphere 1976", "Physics", "not launch-day weather"),
    ("Lunar module mass at launch, descent orbit burn", "Flight data", "LM, PK"),
    ("Descent engine: 10,500 lbf rated, 92.5 % full throttle, 305 s", "Engine rating", "TN"),
    ("State at ignition: 48,814 ft, 5,560 ft/s", "NASA plan", "TM table I"),
    ("High gate state, typical phase lengths and terminus", "NASA design", "TM table I, DL"),
    ("Descent rates and forward speeds below 400 ft", "Crew callouts", "AL"),
    ("Moon: point-mass gravity, spin, no air", "Physics", ""),
]
outputs = [
    "Combustion temperature, exhaust speed, vacuum Isp and exit area of every engine (chemical equilibrium)",
    "Thrust as the air thins, every 20 ms",
    "The whole ascent: altitude, speed, flight path, range, dynamic pressure, Mach, acceleration, mass",
    "When the S-IVB guidance shuts the engine down, and the orbit it leaves the vehicle in",
    "The whole braking phase: steering and throttle from Apollo's guidance law, every 20 ms",
    "When the throttle comes down, when high gate is reached, where and when Eagle touches down",
    "Propellant burnt and left, delta-V, the mass on the Moon",
]
in_table = table([("Given", ""), ("Kind", "kind"), ("Source", "src")],
                 [(esc(a), f'<span class="chip">{esc(b)}</span>', esc(c)) for a, b, c in inputs])

eng = [("F-1 (S-IC)", "RP-1 / LOX", CA["F1"], "304 s published vacuum Isp"),
       ("J-2 (S-II, mixture 5.5)", "LH2 / LOX", CA["J2_hi"], "from the S-II's measured thrust and propellant use"),
       ("J-2 (S-IVB)", "LH2 / LOX", CA["J2_s4"], "from the S-IVB's measured thrust and flow"),
       ("Descent engine (LM)", "Aerozine 50 / N2O4", CL["dps"], "305 s specification [TN]")]
eng_table = table([("Engine", ""), ("Propellants", ""), ("Chamber temperature", "num"), ("Ideal vacuum Isp", "num"),
                   ("Delivered vacuum Isp", "num"), ("Efficiency", "num"), ("Anchored to", "src")],
                  [(esc(n), esc(p), f(float(d["Tc"]), 0) + " K", f(float(d["Isp_vac_ideal"]), 1) + " s",
                    f(float(d["Isp_vac"]), 1) + " s", f(100 * float(d["efficiency"]), 1) + " %", esc(a)) for n, p, d, a in eng])

src_list = "".join(
    f'<li><b>{k}</b> ' + (f'<a href="{esc(s["url"])}">{esc(s["title"])}</a>' if s["url"] else esc(s["title"])) + "</li>"
    for k, s in REF["sources"].items())

tiles = [
    ("Orbit altitude", f"{RA['h_orbit']:.1f} km", f"NASA {cp['h_orbit'][4]} km"),
    ("Orbit speed, Earth-fixed", f"{RA['vE_orbit']:,.1f} m/s", f"NASA {cp['vE_orbit'][4]:,} m/s"),
    ("Mass in orbit", f"{RA['m_orbit'] / 1000:.1f} t", f"NASA {cp['m_orbit'][4]:.1f} t"),
    ("Braking-phase speed vs NASA plan", f"within {worst_v:.1f} %", "six checkpoints, 26 s to 506 s"),
    ("Throttle recovery", f"{RL['t_throttle']:.0f} s", "flight 386 s, plan 384 s"),
    ("Touchdown after ignition", mmss(t_td), "flight 12:34.7"),
]
tiles_html = "".join(f'<div class="tile"><div class="tl">{esc(a)}</div><div class="tv">{esc(b)}</div><div class="ts">{esc(c)}</div></div>'
                     for a, b, c in tiles)

misses = [
    f"Max dynamic pressure is {abs(pct(RA['q_max'], 35.2)):.0f} % low and {83.0 - RA['T_qmax']:.0f} s early: standard atmosphere and a generic drag curve, not the day's weather or Saturn V wind-tunnel data.",
    f"At S-II/S-IVB separation the flight path angle is {RA['fpa_sep2']:.2f}° against NASA's 0.61°.",
    f"The S-IVB cuts off {RA['T_eco'] - 699.33:.1f} s late.",
    f"The braking phase dips early: {at(lf, 'alt', 384):,.0f} ft at 384 s against the plan's 24,639 ft. Apollo 11's own guidance targets were loaded into the computer before flight and are not in these sources, so the canvas aims at NASA's published high gate instead.",
    f"Throttle recovery comes {RL['t_throttle'] - 386:.0f} s late and the approach runs about 20 s ahead, so the sim's crew phase starts early and Eagle lands {754.7 - t_td:.0f} s sooner than the real one.",
    f"Eagle had about 770 lb of propellant left; the sim has {RL['prop_left']:,.0f} lb. That gap is {abs(pct(used_sim, used_flight)):.1f} % of the propellant burnt.",
]
miss_html = "".join(f"<li>{esc(m)}</li>" for m in misses)


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


FONTS = os.path.join(ROOT, "simblocks", "app", "fonts")
font_css = "".join(
    f'@font-face{{font-family:"{fam}";src:url(data:font/woff2;base64,{b64(os.path.join(FONTS, fn))}) format("woff2");font-weight:{wt}}}'
    for fam, fn, wt in (("Plex Sans", "ibm-plex-sans-latin-400-normal.woff2", 400),
                        ("Plex Sans", "ibm-plex-sans-latin-600-normal.woff2", 600),
                        ("Plex Mono", "ibm-plex-mono-latin-400-normal.woff2", 400)))

CSS = """
:root{--bg:#04060a;--panel:#0a0e15;--line:#1b2331;--ink:#eef2f8;--ink2:#a9b4c4;--ink3:#6f7c90;--nasa:#ff8a3d;--sim:#f2f5fa;
--flight:#5ad1ff;--call:#ffd166;--ok:#5fe0a4;--warn:#ffd166;--bad:#ff6b6b;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Plex Sans",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
a{color:var(--flight)}
.wrap{max-width:1180px;margin:0 auto;padding:0 max(20px,env(safe-area-inset-left)) 80px}
header.hero{position:relative;overflow:hidden;border-bottom:1px solid var(--line);
 background:radial-gradient(1200px 500px at 85% -10%,#18233a 0%,rgba(4,6,10,0) 60%),var(--bg)}
.stars{position:absolute;inset:0;pointer-events:none;opacity:.55}
.hero .wrap{position:relative;padding-top:72px;padding-bottom:56px}
.kicker{font:12px/1.4 "Plex Mono",ui-monospace,monospace;letter-spacing:.22em;text-transform:uppercase;color:var(--ink3)}
h1{font-weight:600;font-size:clamp(44px,8vw,96px);line-height:.95;letter-spacing:-.02em;margin:14px 0 18px}
h1 span{display:block;font-weight:400;font-size:.36em;letter-spacing:0;color:var(--ink2);margin-top:14px;line-height:1.25}
.lede{max-width:760px;color:var(--ink2);font-size:18px;margin:0}
.lede b{color:var(--ink);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:1px;background:var(--line);
 border:1px solid var(--line);margin-top:40px}
.tile{background:var(--panel);padding:16px 18px;min-width:0}
.tl{font:11px/1.4 "Plex Mono",ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.tv{font-size:clamp(21px,5.6vw,28px);font-weight:600;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.ts{font-size:13px;color:var(--nasa)}
section{padding-top:64px}
h2{font-weight:600;font-size:clamp(26px,3.6vw,38px);letter-spacing:-.01em;margin:6px 0 10px}
.sec-lede{color:var(--ink2);max-width:780px;margin:0 0 26px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:18px}
.card{margin:0;background:var(--panel);border:1px solid var(--line);padding:18px 18px 10px;min-width:0}
.card.wide{grid-column:1/-1;margin-bottom:18px}
figcaption h3{margin:0;font-size:16px;font-weight:600}
figcaption p{margin:4px 0 10px;color:var(--ink2);font-size:14px;line-height:1.5}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--ink2);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.lg-line{width:20px;height:2px;display:inline-block}
.lg-dash{width:20px;height:0;border-top:2px dashed;display:inline-block}
.lg-dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.lg-dia{width:8px;height:8px;transform:rotate(45deg);display:inline-block}
.lg-ring{width:11px;height:11px;border-radius:50%;border:2px solid;display:inline-block}
.lg-ev{width:0;height:12px;border-left:1px dashed #46536a;display:inline-block}
.plot{position:relative;overflow-x:auto;overflow-y:hidden}
.plot svg.chart{min-width:460px}
.card.wide .plot svg.chart{min-width:720px}
.yl{font:11px/1.4 "Plex Mono",ui-monospace,monospace;color:#7d8a9d;margin:4px 0 2px;letter-spacing:.04em}
svg.chart{display:block;overflow:visible}
svg .tk{font:11px "Plex Mono",ui-monospace,monospace;fill:#7d8a9d}
svg .ax{font:11px "Plex Mono",ui-monospace,monospace;fill:#7d8a9d;letter-spacing:.04em}
svg .ev{font:10.5px "Plex Mono",ui-monospace,monospace;fill:#8e9bb0}
svg .band{font:10.5px "Plex Mono",ui-monospace,monospace;fill:#55627a;text-anchor:middle;letter-spacing:.08em;text-transform:uppercase}
.tip{position:absolute;pointer-events:none;background:#0f141d;border:1px solid #2a3446;padding:8px 10px;font:12px/1.5 "Plex Mono",ui-monospace,monospace;
 white-space:nowrap;z-index:3;box-shadow:0 8px 28px rgba(0,0,0,.5)}
.tip .t{color:var(--ink3)}
.tip .r{display:flex;gap:8px;align-items:center}
.tip i{width:8px;height:8px;border-radius:50%;display:inline-block}
.tw{overflow-x:auto;border:1px solid var(--line);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;font:11px/1.4 "Plex Mono",ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);
 padding:12px 14px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 14px;border-bottom:1px solid #121925;vertical-align:top}
tr:last-child td{border-bottom:0}
.num{text-align:right;white-space:nowrap}
.src{color:var(--ink3);font-size:13px}
.mut{color:var(--ink3)}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.chip{font:11px/1.4 "Plex Mono",ui-monospace,monospace;letter-spacing:.06em;border:1px solid #2a3446;padding:2px 8px;white-space:nowrap;color:var(--ink2)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:18px}
.box{background:var(--panel);border:1px solid var(--line);padding:18px 22px}
.box h3{margin:0 0 8px;font-size:16px}
.box ul{margin:0;padding-left:20px;color:var(--ink2)}
.box li{margin:6px 0}
pre{background:#080b11;border:1px solid var(--line);padding:16px 18px;overflow-x:auto;font:13px/1.6 "Plex Mono",ui-monospace,monospace;color:#cfd8e6}
.timeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,150px),1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:8px 0 26px}
.timeline div{background:var(--panel);padding:12px 14px;min-width:0}
.timeline b{display:block;font:11px/1.4 "Plex Mono",ui-monospace,monospace;letter-spacing:.12em;color:var(--ink3);text-transform:uppercase}
.timeline span{font-size:14px;color:var(--ink2)}
.next{border:1px dashed #2a3446;padding:16px 20px;color:var(--ink2)}
footer{border-top:1px solid var(--line);margin-top:80px;padding-top:24px;color:var(--ink3);font-size:13px}
ol.src-list{padding-left:20px;color:var(--ink2);font-size:14px}
ol.src-list li{margin:6px 0}
"""

JS = r"""
const D = JSON.parse(document.getElementById('chart-data').textContent);
const fmt = (v, u) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) >= 10 ? 1 : 2))) + (u ? ' ' + u : '');
function interp(xs, ys, x){ if (x < xs[0] || x > xs[xs.length - 1]) return null; let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1){ const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const f = xs[hi] === xs[lo] ? 0 : (x - xs[lo]) / (xs[hi] - xs[lo]); return ys[lo] + f * (ys[hi] - ys[lo]); }
D.forEach(c => {
  const svg = document.getElementById(c.id); if (!svg) return;
  const hit = svg.querySelector('.hit'); if (!hit) return;
  const tip = svg.parentElement.querySelector('.tip');
  const NS = 'http://www.w3.org/2000/svg';
  const rule = document.createElementNS(NS, 'line'); rule.setAttribute('y1', c.mT); rule.setAttribute('y2', c.H - c.mB);
  rule.setAttribute('stroke', '#55627a'); rule.style.display = 'none'; svg.insertBefore(rule, hit);
  const dots = c.lines.map(l => { const d = document.createElementNS(NS, 'circle'); d.setAttribute('r', 4); d.setAttribute('fill', l.color);
    d.setAttribute('stroke', '#04060a'); d.setAttribute('stroke-width', 1.5); d.style.display = 'none'; svg.insertBefore(d, hit); return d; });
  const X = x => c.mL + (x - c.x0) / (c.x1 - c.x0) * (c.W - c.mL - c.mR);
  const Y = y => c.mT + (c.y1 - y) / (c.y1 - c.y0) * (c.H - c.mT - c.mB);
  const move = ev => {
    const r = svg.getBoundingClientRect(); const sx = (ev.clientX - r.left) / r.width * c.W;
    const x = c.x0 + (sx - c.mL) / (c.W - c.mL - c.mR) * (c.x1 - c.x0);
    rule.setAttribute('x1', X(x)); rule.setAttribute('x2', X(x)); rule.style.display = '';
    tip.innerHTML = ''; const head = document.createElement('div'); head.className = 't';
    head.textContent = fmt(x, c.xunit); tip.appendChild(head);
    c.lines.forEach((l, i) => { const y = interp(l.x, l.y, x); if (y === null) { dots[i].style.display = 'none'; return; }
      dots[i].setAttribute('cx', X(x)); dots[i].setAttribute('cy', Y(y)); dots[i].style.display = '';
      const row = document.createElement('div'); row.className = 'r'; const s = document.createElement('i'); s.style.background = l.color;
      const t = document.createElement('span'); t.textContent = l.name + '  ' + fmt(y, c.yunit); row.append(s, t); tip.appendChild(row); });
    const near = c.points.filter(p => Math.abs(X(p.x) - sx) < 10);
    near.forEach(p => { const row = document.createElement('div'); row.className = 'r'; const s = document.createElement('i'); s.style.background = p.color;
      const t = document.createElement('span'); t.textContent = p.label + '  ' + fmt(p.y, c.yunit); row.append(s, t); tip.appendChild(row); });
    tip.hidden = false; const sc = svg.parentElement, vis = sc.clientWidth, sl = sc.scrollLeft;
    const bw = svg.clientWidth, tw = tip.offsetWidth, px = (X(x) / c.W) * bw;
    let left = px + 14; if (left + tw > sl + vis) left = px - tw - 14;
    tip.style.left = Math.max(sl, Math.min(sl + vis - tw, left)) + 'px'; tip.style.top = '6px';
  };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerleave', () => { tip.hidden = true; rule.style.display = 'none'; dots.forEach(d => d.style.display = 'none'); });
});
// a star field drawn once
(function(){ const s = document.querySelector('.stars'); if (!s) return; let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const NS = 'http://www.w3.org/2000/svg'; for (let i = 0; i < 160; i++){ const c = document.createElementNS(NS, 'circle');
  c.setAttribute('cx', (rnd() * 100).toFixed(2) + '%'); c.setAttribute('cy', (rnd() * 100).toFixed(2) + '%'); c.setAttribute('r', (rnd() * 1.1 + 0.2).toFixed(2));
  c.setAttribute('fill', '#cfd8e6'); c.setAttribute('opacity', (rnd() * 0.7 + 0.15).toFixed(2)); s.appendChild(c); } })();
"""

leg_asc = legend([("REDblocks", COL["sim"], "line"), ("NASA measured", COL["nasa"], "dot"), ("staging and cutoff", "", "ev")])
leg_land = legend([("REDblocks, as flown", COL["sim"], "line"), ("REDblocks, automatic landing", COL["auto"], "dash"),
                   ("NASA premission plan", COL["nasa"], "dot"), ("Apollo 11 flight", COL["flight"], "ring")])
leg_final = legend([("REDblocks, as flown", COL["sim"], "line"), ("REDblocks, automatic landing", COL["auto"], "dash"),
                    ("crew callout", COL["call"], "diamond"), ("Apollo 11 flight", COL["flight"], "ring")])
leg_track = legend([("REDblocks, as flown", COL["sim"], "line"), ("REDblocks, automatic landing", COL["auto"], "dash")])


def with_legend(card_html, leg):
    return card_html.replace("</figcaption>", "</figcaption>" + leg, 1)


timeline_html = "".join(f"<div><b>{esc(a)}</b><span>{esc(b)}</span></div>" for a, b in (
    ("0:00", "Saturn V first motion, 16 July 1969"),
    ("2:42", "S-IC staging, 67 km up"),
    ("9:09", "S-II staging, 187 km"),
    ("11:49", "Earth parking orbit"),
    ("102:33 GET", "Eagle lights its descent engine"),
    ("102:45 GET", "Touchdown, Sea of Tranquility"),
))

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Apollo 11 in REDblocks</title>
<meta name="description" content="The Apollo 11 ascent and lunar landing simulated in REDblocks and checked against NASA's flight records.">
<style>{font_css}{CSS}</style></head>
<body>
<header class="hero"><svg class="stars" width="100%" height="100%" aria-hidden="true"></svg><div class="wrap">
<div class="kicker">REDblocks showcase · two canvases · checked against NASA</div>
<h1>Apollo 11<span>Launch to orbit. Powered descent to the Moon. Simulated.</span></h1>
<p class="lede">Two REDblocks canvases fly the mission from <b>rocket chemistry and Newton's laws</b>: the Saturn V from the pad to Earth orbit,
and the lunar module Eagle from 260 nautical miles out to touchdown, steered by <b>the same guidance law the Apollo computer flew</b>.
Every curve below is the sim. Every dot is NASA's.</p>
<div class="tiles">{tiles_html}</div>
</div></header>
<main class="wrap">
<section id="ascent"><div class="kicker">Part 1 · Saturn V AS-506</div><h2>The climb to orbit</h2>
<p class="sec-lede">Five F-1s, then five J-2s, then one. The engines come from equilibrium combustion chemistry; the masses, thrust levels and event
times come from Marshall's flight evaluation report. One number was tuned: the pitch-over rate. Everything after that is a gravity turn and
guidance, and the vehicle arrives in the same orbit as Apollo 11.</p>
<div class="timeline">{timeline_html}</div>
{with_legend(asc_track, leg_asc)}
<div class="grid">{"".join(with_legend(c, leg_asc) for c in asc_charts)}</div>
<h3 style="margin:34px 0 12px">Every ascent checkpoint</h3>
{asc_table}
</section>
<section id="landing"><div class="kicker">Part 2 · Lunar module Eagle</div><h2>Powered descent to touchdown</h2>
<p class="sec-lede">The descent engine burns Aerozine 50 with nitrogen tetroxide. Apollo's quadratic guidance law, solved every 20 ms in the
landing-site frame, steers it and throttles it: 10 % for 26 seconds, full throttle, then <b>throttle recovery</b> when the law asks for less
than 57 %. Below 400 ft the sim flies the descent rates Aldrin called out, as Armstrong did. The dashed line is the automatic landing NASA planned
had nobody taken over.</p>
{with_legend(land_track, leg_track)}
<div class="grid">{"".join(with_legend(c, leg_land) for c in land_charts)}</div>
{with_legend(land_final, leg_final)}
<div class="grid">{with_legend(land_prop, leg_land)}{with_legend(land_cmd, legend([("guidance asks for", COL["call"], "line"), ("throttle", COL["sim"], "line"), ("NASA", COL["nasa"], "dot")]))}</div>
<h3 style="margin:34px 0 12px">Against NASA's descent plan (the automatic landing)</h3>
{plan_table}
<h3 style="margin:34px 0 12px">Against the flight</h3>
{key_table}
</section>
<section id="engines"><div class="kicker">Under the hood</div><h2>Engines from chemistry</h2>
<p class="sec-lede">Each engine is a REDblocks <code>rocket_engine</code> block: propellants burnt to chemical equilibrium in the chamber,
then expanded through the nozzle with the chemistry shifting as it cools (the NASA CEA method). The block gives the ideal performance;
one published rating sets how far below ideal the real engine sits. The descent engine's chamber temperature matches Cantera's to the
hundredth of a kelvin.</p>
{eng_table}
</section>
<section id="honest"><div class="kicker">What the sim is given, and where it misses</div><h2>Measured vs simulated</h2>
<div class="two"><div class="box"><h3>The canvases are given</h3>{in_table}</div>
<div class="box"><h3>REDblocks works out</h3><ul>{"".join(f"<li>{esc(x)}</li>" for x in outputs)}</ul>
<h3 style="margin-top:22px">Where it misses</h3><ul>{miss_html}</ul></div></div>
<p class="next" style="margin-top:22px"><b>Left out on purpose:</b> the translunar injection burn, the three-day coast and lunar orbit
insertion. The page shows the two powered flights where the physics is hardest: climbing out of Earth's atmosphere, and landing on the Moon
under guidance.</p>
</section>
<section id="run"><div class="kicker">Run it yourself</div><h2>Two files, one command each</h2>
<pre>pip install .                                              # from the scripts folder
simblocks run examples/showcase/apollo11/apollo11_1_ascent.yaml
simblocks run examples/showcase/apollo11/apollo11_2_landing.yaml
simblocks set examples/showcase/apollo11/apollo11_2_landing.yaml crew=0    # the automatic landing
python examples/showcase/apollo11/build_page.py            # rebuild this page
simblocks app                                              # open either canvas in the node editor</pre>
</section>
<section id="sources"><div class="kicker">Sources</div><h2>Where NASA's numbers come from</h2>
<ol class="src-list">{src_list}</ol>
<p class="mut" style="font-size:13px">Built {esc(__import__('datetime').date.today().isoformat())} by build_page.py from the canvases in this folder.
Ascent energy ledger closes to {RA['_ledger_rel']:.1e}; landing to {RL['_ledger_rel']:.1e}. Fonts: IBM Plex (SIL Open Font License).</p>
</section>
<footer>REDblocks · Apollo 11 showcase</footer>
</main>
<script type="application/json" id="chart-data">{json.dumps(CHART_DATA, separators=(",", ":"))}</script>
<script>{JS}</script>
</body></html>
"""
out = os.path.join(HERE, "apollo11.html")
open(out, "w").write(page)
print("wrote", out, f"{len(page) / 1e6:.2f} MB")

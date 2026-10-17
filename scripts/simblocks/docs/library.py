"""Block library: one searchable HTML page generated from the registry, so it never drifts from the code.
Run:  simblocks library [out.html]     or     python -m simblocks.docs.library out.html
Each entry: what it does, inputs, outputs, how to use it in a system file, and a small diagram."""
import ast
import html
import inspect
import json
import os
import sys
import textwrap

from . import figures as FG

# kernel blocks that the 1d compiler already drives through an element type
KERNEL_ELEMENT = {
    "tabular_return": "elastoplastic", "penalty_kelvin_voigt": "kelvin_voigt", "penalty_linear": "spring",
    "hunt_crossley": "hertz", "hertz_sphere_force": "hertz", "coulomb_tanh": "friction", "stribeck_mu": "friction",
    "simscape_friction": "friction", "bar_windage_torque": "drag", "esc_supply": "drive", "lipo_ocv_k": "drive",
    "ecm_constant_power": "drive", "bearing_drag_torque": "friction",
}

ELEMENTS = [
    dict(name="spring", law="F = k s", fields="k (stiffness)",
         what="Linear spring along a measure. Two-sided: pushes and pulls.",
         yaml="link:\n  type: spring\n  measure: a - b\n  k: 200 kN/m"),
    dict(name="dashpot", law="F = c ds/dt", fields="c (damping)",
         what="Viscous damper along a measure.", yaml="damp:\n  type: dashpot\n  measure: a - b\n  c: 40 N s/m"),
    dict(name="kelvin_voigt", law="F = k s + c ds/dt (one_sided: never pulls)", fields="k, c, one_sided",
         what="Spring and damper in parallel. With one_sided it is a contact that only pushes, e.g. a floor or tire.",
         yaml="floor:\n  type: kelvin_voigt\n  measure: chassis - floor\n  k: 200 kN/m\n  c: \"= $c_floor\"\n  one_sided: true"),
    dict(name="elastoplastic", law="tabular_return: elastic k, then yield force from a hardening table",
         fields="table (k, acc_tab, F_tab from a builder), one_sided, break_at",
         what="Spring that yields and keeps a permanent set. Tables come from builders such as beam_plastic_spring "
              "or plastic_indentation_table. break_at removes it after a set amount (punch-through, tear-off).",
         yaml="shaft:\n  type: elastoplastic\n  measure: hub - frame\n  table: $shaft      # from derived: {block: beam_plastic_spring, ...}"),
    dict(name="hertz", law="F = K s^n (1 + alpha ds/dt), compression only", fields="K, n (1.5), alpha (0)",
         what="Nonlinear contact (Hertz). alpha adds Hunt-Crossley or Flores damping for a target restitution.",
         yaml="ball:\n  type: hertz\n  measure: ball - plate\n  K: \"= 4/3 * $Estar * sqrt($R)\"\n  n: 1.5"),
    dict(name="friction", law="F = F_c tanh(v / v_eps) (+ Stribeck peak F_s at v_s)", fields="F_c, v_eps, F_s, v_s",
         what="Regularized Coulomb friction on the measure's velocity, with optional static peak.",
         yaml="skid:\n  type: friction\n  measure: sled\n  F_c: \"= 0.2 * $m * 9.81\"\n  v_eps: 5 mm/s"),
    dict(name="drag", law="F = K v |v|", fields="K",
         what="Quadratic drag: windage on a spinning rotor, air drag on a body.",
         yaml="windage:\n  type: drag\n  measure: rotor\n  K: $K_wind      # from bar_windage_torque at omega = 1"),
    dict(name="force", law="F = P (constant), a standard shape F(t) (pulse), or F(t) / F(s) from a CSV table", fields="P, or pulse: {kind, ...}, or csv: {file, against, x_unit, y_unit, interpolation, x_col, y_col}",
         what="A push along the measure. Constant (P), or following a table: against time (a measured or designed load "
              "history) or against the measure itself (a load-deflection curve). The table comes from a CSV file (path "
              "relative to the model's folder) or inline x / y lists; you give the units of both columns; rows are joined "
              "linearly (default), held (step) or by a monotone cubic with no overshoot (smooth, PCHIP); the end values "
              "hold outside the table. Or a standard shape (pulse): half_sine, haversine, triangle, rectangle, trapezoid, "
              "ramp, step, sine, square_wave, sweep (chirp) or decay, each from a few numbers (peak, duration, frequency...).",
         yaml="shock:\n  type: force\n  measure: instrument\n  csv: {file: data/shock.csv, against: time, x_unit: ms, y_unit: kN, interpolation: linear}"),
    dict(name="drive", law="DC motor + current-limited ESC + LiPo 1-RC pack", fields="kv, R, I_max, gear, cells, R0, R1, C1, capacity, throttle, J_rotor",
         what="Powers one rotation body. Tracks motor current, battery current, bus voltage and state of charge.",
         yaml="motor:\n  type: drive\n  measure: rotor\n  kv: 1100 rpm/V\n  R: 35 mohm\n  I_max: 60 A\n  gear: 1.5\n  cells: 5\n  R0: 0.15 ohm\n  capacity: 450 mAh"),
    dict(name="pin", law="f = k g + c dg/dt on the gap vector g between two points (2d / 3d)", fields="k, c",
         what="2D and 3D. Holds a point on one body to a point on another with a stiff spring-damper in every "
              "direction, so the bodies turn freely about it: a hinge in 2D, a ball joint in 3D (two pins along a line "
              "make a 3D hinge). Use k well above the other stiffnesses; the solver's time step follows it.",
         yaml="hinge:\n  type: pin\n  measure: [chassis.root, face.root]\n  k: 5e7 N/m\n  c: 200 N s/m"),
    dict(name="impact", law="rigid contact: an impulse sets the separating speed to e x the approach speed; Coulomb friction mu",
         fields="e (restitution 0-1), mu, v_rest",
         what="A hard hit with no stiffness to choose. When the measure (a contact, or any measure where s > 0 means "
              "overlap) would close within the step, an impulse reverses the approach speed times e, and friction up to mu "
              "times that impulse acts along the surface (sticking if it can). Every joule removed is booked as dissipated energy "
              "(the dissipated output; its history is the running total). A contact held closed (resting, or pressed by a motor) does not "
              "bounce and books only the friction of real sliding (none while it sticks). Use it when you care where things go and how much energy is lost, not the force profile: "
              "the force output is impulse / dt. For peak forces, dents and deformation use a penalty contact "
              "(kelvin_voigt, hertz, elastoplastic) instead. See Help > Contact and joint methods.",
         yaml="bounce:\n  type: impact\n  e: 0.6\n  mu: 0.4\n  measure: {contact: [ball.outline, floor.top]}"),
    dict(name="rope", law="F = k (L - length) + c dL/dt when L > length, else 0 (2d / 3d)", fields="k, c, length",
         what="2D and 3D. Pulls two points together only once the distance between them exceeds its length: slack "
              "otherwise. A tether, cable or strap.",
         yaml="tether:\n  type: rope\n  measure: {distance: [post.top, load.hook], length: 1.2 m}\n  k: 50 kN/m\n  c: 20 N s/m"),
    dict(name="weld", law="pin + rotation lock (2d / 3d)", fields="rigid, or k, c, k_rot, c_rot",
         what="2D and 3D. Joins two bodies so they move and turn as one. rigid: true makes it exact (projected constraints: "
              "no stiffness to choose, no time-step cost). Otherwise a stiff spring-damper in translation (k, c) and "
              "rotation (k_rot, c_rot).",
         yaml="bolted:\n  type: weld\n  rigid: true\n  measure: [frame.mount, motor.base]"),
    dict(name="hinge", law="pin, plus (3D) a lock of the turns across its axis; optional torsion spring k_twist", fields="rigid, axis (3D), k_twist, c_twist, or k, c, k_rot",
         what="2D and 3D. Two points held together, free to turn about the hinge axis (in 2D the plane's normal). "
              "rigid: true makes it exact. k_twist / c_twist add a torsion spring and damper about the axis (a spring-loaded "
              "joint, bearing drag). The force output is the reaction the hinge carries; drift is how far its points ever "
              "got apart before being put back (rigid).",
         yaml="elbow:\n  type: hinge\n  rigid: true\n  axis: y\n  k_twist: 60 N m/rad\n  measure: [upper.e, fore.e]"),
    dict(name="slider", law="B's point held on a line fixed in A, free along it", fields="axis, rigid, or k, c; k_rot, friction",
         what="2D and 3D. A guide rail: B's point may move only along the axis (given at the start, then turning with A). "
              "rigid: true makes it exact. k_rot (any value when rigid) also stops B turning relative to A. friction adds "
              "Coulomb drag along the rail. Its measure output is how far B has slid.",
         yaml="carriage:\n  type: slider\n  rigid: true\n  axis: x\n  k_rot: 0\n  measure: [rail.o, cart.p]"),
    dict(name="beam", law="co-rotational beam: N = EA/L0 (L - L0), end moments from EI (2D: one plane; 3D: two planes and GJ torsion)",
         fields="section or A, I (Iy, J), E (G), material, density or mass, segments, damping, orient, hinge, end_mass",
         what="2D and 3D. A deformable beam between two points: stretches, bends and (3D) twists, through any large "
              "rotation. Give a material and a section block (or E, A, I, J), and segments: n to split it into n pieces "
              "with node bodies carrying its mass (so it vibrates and whips). hinge: auto (or a moment table) forms "
              "plastic hinges at Mp = sy Z. orient (3D) is the section's h direction. Outputs: moment, torque, shear, "
              "hinge_set.",
         yaml="arm:\n  type: beam\n  measure: [hub.root, motor.mount]\n  section: $tube\n  E: 70 GPa\n  density: 1550 kg/m^3\n  segments: 6"),
    dict(name="contact_friction", law="F = mu |F_n| tanh(v_t / v_eps), along the contact surface (2d / 3d)", fields="contact, mu, v_eps",
         what="2D and 3D. Sliding friction on a contact element: rides on that contact's normal force and acts along "
              "its surface, so it stops when the contact separates. Rolling, skidding and scraping come out of it.",
         yaml="grip:\n  type: contact_friction\n  contact: tyre_on_floor\n  mu: 0.8\n  v_eps: 1 mm/s"),
]


# older one-case blocks the general cantilever covers: kept for files that use them, left out of the Add menu
SUPERSEDED = {"cantilever_end_load": "cantilever", "cantilever_point_at": "cantilever", "cantilever_uniform": "cantilever",
              "cantilever_end_moment": "cantilever", "cantilever_tip_deflection_point_at": "cantilever"}
_PD = None


def _param_docs():
    """One short description per block argument (docs/param_docs.json), shown in the editor and the library."""
    global _PD
    if _PD is None:
        import json, os
        try:
            _PD = json.load(open(os.path.join(os.path.dirname(__file__), "param_docs.json"), encoding="utf-8"))
        except Exception:
            _PD = {}
    return _PD


def _el_fig(name):
    return FG.place_labels("el-" + name, _el_fig_raw(name))


def _el_fig_raw(name):
    a = FG._box(8, 30, 34, 30, "A") + FG._box(118, 30, 34, 30, "B")
    mid = {"spring": FG._spring(42, 118, 45),
           "dashpot": '<line x1="42" y1="45" x2="70" y2="45" stroke="currentColor" stroke-width="1.4"/>'
                      '<path d="M70,35 V55 M70,35 H92 M70,55 H92" fill="none" stroke="currentColor" stroke-width="1.4"/>'
                      '<line x1="84" y1="38" x2="84" y2="52" stroke="currentColor" stroke-width="2"/>'
                      '<line x1="84" y1="45" x2="118" y2="45" stroke="currentColor" stroke-width="1.4"/>',
           "kelvin_voigt": FG._spring(42, 118, 38, 5, 4) + '<line x1="42" y1="54" x2="70" y2="54" stroke="currentColor"/>'
                           '<path d="M70,48 V60 M70,48 H90 M70,60 H90" fill="none" stroke="currentColor"/>'
                           '<line x1="82" y1="50" x2="82" y2="58" stroke="currentColor" stroke-width="2"/>'
                           '<line x1="82" y1="54" x2="118" y2="54" stroke="currentColor"/>',
           "elastoplastic": FG._spring(42, 88, 45, 4) + '<rect x="88" y="38" width="14" height="14" style="fill:var(--soft)" stroke="currentColor"/>'
                            '<line x1="102" y1="45" x2="118" y2="45" stroke="currentColor"/>' + FG._ground(84, 106, 54),
           "hertz": '<line x1="42" y1="45" x2="78" y2="45" stroke="currentColor"/>' + FG._circle(86, 45, 8) +
                    '<line x1="94" y1="45" x2="118" y2="45" stroke="currentColor"/>',
           "friction": "",
           "drag": '<path d="M48,32 q8,6 0,12 q-8,6 0,12" fill="none" style="stroke:var(--acc)"/>'
                   '<path d="M64,32 q8,6 0,12 q-8,6 0,12" fill="none" style="stroke:var(--acc)"/>'
                   '<line x1="80" y1="45" x2="118" y2="45" stroke="currentColor"/>',
           "force": FG._arrow(60, 45, 113, 45),
           "drive": FG._box(52, 32, 40, 26, "M") + '<line x1="92" y1="45" x2="118" y2="45" stroke="currentColor"/>',
           "pin": '<line x1="42" y1="45" x2="76" y2="45" stroke="currentColor" stroke-width="1.4"/>' + FG._circle(80, 45, 5, soft=False) +
                  '<line x1="84" y1="45" x2="118" y2="45" stroke="currentColor" stroke-width="1.4"/>',
           "contact_friction": "",
           "impact": "", "rope": '<path d="M42,45 q19,14 38,0 q19,-14 38,0" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 3"/>',
           "weld": '<rect x="42" y="38" width="76" height="14" style="fill:var(--soft)" stroke="currentColor"/>',
           "hinge": '<line x1="42" y1="45" x2="74" y2="45" stroke="currentColor" stroke-width="1.4"/>' + FG._circle(80, 45, 7, soft=False) +
                    '<line x1="86" y1="45" x2="118" y2="45" stroke="currentColor" stroke-width="1.4"/>',
           "slider": '<line x1="42" y1="58" x2="118" y2="58" stroke="currentColor" stroke-width="2"/>' +
                     '<rect x="66" y="44" width="28" height="12" style="fill:var(--soft)" stroke="currentColor"/>',
           "beam": '<path d="M42,45 C70,45 90,52 118,60" fill="none" stroke="currentColor" stroke-width="3"/>'}[name]
    if name == "impact":                         # ball dropping onto a surface, with a bounce arrow
        s = (FG._box(14, 58, 132, 15, "ground") + FG._circle(60, 38, 12) + FG._arrow(60, 8, 60, 24) +
             FG._arrow(96, 54, 112, 28) + FG._text(30, 88, "hit: speed out = e x speed in"))
        return f'<symbol id="el-{name}" viewBox="0 0 160 90">{s}</symbol>'

    if name == "contact_friction":                 # block on a surface: normal force down, friction along it
        s = (FG._box(14, 54, 132, 16, "B") + FG._box(56, 28, 44, 26, "A") + FG._arrow(78, 14, 78, 27) + FG._arrow(56, 48, 34, 48) +
             FG._pt(78, 54) + FG._text(26, 84, "friction = mu x contact force"))
        return f'<symbol id="el-{name}" viewBox="0 0 160 90">{s}</symbol>'
    if name == "pin":
        s = FG._box(8, 30, 34, 30, "A") + FG._box(118, 30, 34, 30, "B") + mid + FG._pt(80, 45) + FG._text(30, 84, "two points held together")
        return f'<symbol id="el-{name}" viewBox="0 0 160 90">{s}</symbol>'
    if name == "friction":                      # block A sliding on surface B; friction acts on s = A - B
        s = (FG._box(14, 50, 132, 14, "B") + FG._box(58, 24, 40, 24, "A") + FG._arrow(102, 36, 128, 36) +
             '<line x1="58" y1="49" x2="98" y2="49" stroke="currentColor" stroke-width="2.2" stroke-dasharray="2 2"/>' +
             FG._pt(78, 24) + FG._pt(30, 57) + FG._text(46, 84, "measure s = A - B"))
        return f'<symbol id="el-{name}" viewBox="0 0 160 90">{s}</symbol>'
    if name in ("force", "drag", "drive"):
        a = FG._box(118, 30, 34, 30, "body")
    one = name in ("force", "drag", "drive")
    s = a + mid + ("" if one else FG._pt(42, 45))     # one point per body end: a one-body element has just the one
    s += FG._pt(118, 45) + FG._text(46, 84, "measure s = body" if one else "measure s = A - B")
    return f'<symbol id="el-{name}" viewBox="0 0 160 90">{s}</symbol>'


def _outputs(fn):
    """Read what a block returns from its source (dict keys, tuple names) or its docstring."""
    py = getattr(fn, "py_func", fn)
    doc = inspect.getdoc(py) or ""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(py)))
    except (OSError, TypeError, SyntaxError):
        tree = None
    found = []
    if tree is not None:
        fdef = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
        for node in ast.walk(fdef) if fdef else []:
            if isinstance(node, ast.Return) and node.value is not None:
                v = node.value
                if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "_sec":
                    extra = [k.arg for k in v.keywords if k.arg]
                    found.append("section dict: A, Ix, Iy, c, S, Z, J, tau_coef, kappa" + (", " + ", ".join(extra) if extra else ""))
                elif isinstance(v, ast.Call) and getattr(v.func, "id", "") == "dict":
                    found.append("dict: " + ", ".join(k.arg for k in v.keywords if k.arg))
                elif isinstance(v, ast.Dict):
                    keys = [ast.literal_eval(k) for k in v.keys if isinstance(k, ast.Constant)]
                    if keys:
                        found.append("dict: " + ", ".join(map(str, keys)))
                elif isinstance(v, ast.Tuple):
                    found.append("(" + ", ".join(ast.unparse(e)[:24] for e in v.elts) + ")")
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id[:1].isupper():
                    found.append(f"{v.func.id} object")
                elif isinstance(v, ast.Name):
                    found.append(v.id)
                else:
                    ex = ast.unparse(v)
                    found.append("number: " + (ex if len(ex) <= 70 else ex[:67] + "..."))
    for line in doc.splitlines():
        if line.strip().lower().startswith("returns"):
            found.insert(0, line.strip()[7:].strip(" :"))
            break
    if found and doc and any(l.strip().lower().startswith("returns") for l in doc.splitlines()):
        # the docstring states the result: keep it, plus only the dict keys read from code (no expression guesses)
        found = found[:1] + [f for f in found[1:] if f.startswith("dict:")]
    seen, out = set(), []
    for f in found:
        if f and f not in seen and len(out) < 2:
            seen.add(f); out.append(f)
    return out or ["one value (see description)"]


def _usage(b, params):
    if b.kind == "kernel" and b.name in KERNEL_ELEMENT:
        return (f"# runs inside the loop through the '{KERNEL_ELEMENT[b.name]}' element type\n"
                f"elements:\n  my_{KERNEL_ELEMENT[b.name]}: {{type: {KERNEL_ELEMENT[b.name]}, measure: a - b, ...}}")
    req = [p for p in params if p["default"] is None]
    opt = [p for p in params if p["default"] is not None]
    def val(p):
        if p["name"] == "mat":
            return '"4130 45 HRC"'
        if p["name"] in ("material",):
            return '"7075"'
        return f"<{p['name']}>"
    args = ", ".join(f"{p['name']}: {val(p)}" for p in req)
    line = f"derived:\n  my_{b.name[:20]}: {{block: {b.name}" + (f", {args}" if args else "") + "}"
    if opt:
        line += "\n  # optional: " + ", ".join(f"{p['name']} (default {p['default']})" for p in opt)
    if b.kind == "kernel":
        line = "# per-step kernel: no element type drives it yet; call it once in derived:\n" + line
    return line


def entries(include_internal=False):
    from ..registry import REGISTRY, is_public
    out = []
    for b in sorted(REGISTRY.values(), key=lambda x: (x.domain, x.category, x.geometry, x.name)):
        if not include_internal and not is_public(b.name):
            continue
        py = getattr(b.func, "py_func", b.func)
        try:
            sig = inspect.signature(py)
            pd = _param_docs().get(b.name, {})
            params = [dict(name=k, default=(None if p.default is inspect._empty else repr(p.default)), doc=pd.get(k, ""))
                      for k, p in sig.parameters.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
            if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
                params.append(dict(name="*" + next(k for k, p in sig.parameters.items() if p.kind == p.VAR_POSITIONAL),
                                   default=None))
        except (TypeError, ValueError):
            params = []
        doc = inspect.getdoc(py) or b.summary
        out.append(dict(name=b.name, domain=b.domain, category=b.category, path=b.path, kind=b.kind,
                        source=b.source, summary=b.summary, doc=doc, params=params, outputs=_outputs(b.func),
                        usage=_usage(b, params), fig=FG.figure_for(b), tags=list(b.tags),
                        legacy=b.source.startswith("LEGACY"), element=KERNEL_ELEMENT.get(b.name, ""),
                        superseded=SUPERSEDED.get(b.name, "")))
    return out


CSS = r"""
:root{--bg:#0b1018;--panel:#111a26;--panel2:#162131;--fg:#dbe4ef;--fg2:#93a3b8;--fg3:#66768c;--edge:#233247;
--acc:#f2a541;--pt:#3fc1c9;--soft:rgba(147,163,184,.14);--link:#7cc4ff;--warn:#fbbf24;
--display:"Barlow Semi Condensed","Arial Narrow",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
--mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f5f7fa;--panel:#ffffff;--panel2:#eef2f7;
--fg:#16202d;--fg2:#4a5a6e;--fg3:#7a889a;--edge:#d5dde8;--acc:#c46a00;--pt:#0b8a92;--soft:rgba(74,90,110,.10);
--link:#1559a8;--warn:#a16207;color-scheme:light}}
:root[data-theme="light"]{--bg:#f5f7fa;--panel:#ffffff;--panel2:#eef2f7;--fg:#16202d;--fg2:#4a5a6e;--fg3:#7a889a;
--edge:#d5dde8;--acc:#c46a00;--pt:#0b8a92;--soft:rgba(74,90,110,.10);--link:#1559a8;--warn:#a16207;color-scheme:light}
/* layout: header + sticky filter bar, then a single column of element types and a grid of block cards */
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--body);margin:0}
.wrap{max-width:1240px;margin:0 auto;padding-block:28px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px))}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;color:var(--pt);text-transform:uppercase}
h1{font:600 34px/1.1 var(--display);letter-spacing:.01em;margin:8px 0 8px;text-wrap:balance}
h2{font:600 22px/1.2 var(--display);margin:40px 0 6px;letter-spacing:.02em}
.lede{color:var(--fg2);max-width:70ch;margin:0}
.lede code,.note code{font:12.5px var(--mono);color:var(--fg)}
.bar{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);padding-block:12px;
border-bottom:1px solid var(--edge);display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:18px}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
#q{flex:1 1 240px;min-width:0;background:var(--panel);border:1px solid var(--edge);color:var(--fg);
font:14px var(--body);padding:9px 12px;border-radius:6px}
#q:focus-visible,.chip:focus-visible,.copy:focus-visible{outline:2px solid var(--pt);outline-offset:2px}
.chip{background:var(--panel);border:1px solid var(--edge);color:var(--fg2);font:12px var(--mono);padding:6px 10px;
border-radius:999px;cursor:pointer}
.chip[aria-pressed="true"]{background:var(--pt);border-color:var(--pt);color:var(--bg)}
.count{font:12px var(--mono);color:var(--fg3);margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:16px}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:8px;padding:14px;display:flex;flex-direction:column;
gap:10px;min-width:0}
.top{display:flex;gap:12px;align-items:flex-start}
.fig{flex:0 0 132px;height:74px;color:var(--fg2);background:var(--panel2);border-radius:6px}
.fig svg{width:132px;height:74px;display:block}
.fl{font:8px var(--mono);fill:var(--fg2)} .fc{font:9px var(--mono);fill:var(--fg);text-anchor:middle}
.meta{min-width:0}
.name{font:600 14px var(--mono);color:var(--fg);word-break:break-word}
.path{font:11px var(--mono);color:var(--fg3);word-break:break-word}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.tag{font:10.5px var(--mono);padding:2px 7px;border-radius:4px;background:var(--panel2);color:var(--fg2)}
.tag.k-kernel{color:var(--pt)} .tag.k-builder{color:var(--acc)} .tag.legacy{color:var(--warn)}
.sum{margin:0;color:var(--fg)}
.io{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:13px}
.io dt{font:11px var(--mono);color:var(--fg3);text-transform:uppercase;letter-spacing:.08em;padding-top:2px}
.io dd{margin:0;font:12.5px var(--mono);color:var(--fg2);word-break:break-word;min-width:0}
pre{margin:0;background:var(--panel2);border-radius:6px;padding:30px 10px 10px;font:12px/1.45 var(--mono);color:var(--fg);
overflow-x:auto;white-space:pre}
.use{position:relative}
.copy{position:absolute;top:6px;right:6px;background:var(--panel);border:1px solid var(--edge);color:var(--fg2);
font:11px var(--mono);border-radius:4px;padding:3px 7px;cursor:pointer}
details summary{cursor:pointer;font:12px var(--mono);color:var(--link)}
details p{white-space:pre-wrap;font-size:13px;color:var(--fg2);margin:8px 0 0}
.src{font-size:12px;color:var(--fg3)}
.els{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;margin-top:14px}
.el .fig{flex-basis:150px;height:84px}.el .fig svg{width:150px;height:84px}
.law{font:12.5px var(--mono);color:var(--acc)}
.note{color:var(--fg2);font-size:14px;max-width:80ch}
@media (max-width:480px){.grid,.els{grid-template-columns:1fr}.top{flex-direction:column}.fig,.el .fig{flex-basis:auto;width:100%}
.fig svg,.el .fig svg{width:100%}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
"""

JS = r"""
const DATA = __DATA__;
const grid = document.getElementById('grid'), q = document.getElementById('q'), cnt = document.getElementById('count');
let dom = 'all';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function card(b){
  const ins = b.params.length ? b.params.map(p => esc(p.name) + (p.default !== null ? ' = ' + esc(p.default) : '')).join(', ') : 'none';
  return `<article class="card" data-d="${b.domain}">
  <div class="top"><div class="fig" aria-hidden="true"><svg viewBox="0 0 160 90"><use href="#fig-${b.fig}"/></svg></div>
  <div class="meta"><div class="name">${esc(b.name)}</div><div class="path">${esc(b.path)}</div>
  <div class="tags"><span class="tag k-${b.kind}">${b.kind}</span>${b.element ? '<span class="tag">element: ' + b.element + '</span>' : ''}${b.legacy ? '<span class="tag legacy">LEGACY</span>' : ''}</div></div></div>
  <p class="sum">${esc(b.summary)}</p>
  <dl class="io"><dt>in</dt><dd>${ins}</dd><dt>out</dt><dd>${b.outputs.map(esc).join('<br>')}</dd></dl>
  <div class="use"><pre>${esc(b.usage)}</pre><button class="copy" type="button">copy</button></div>
  <details><summary>details and source</summary><p>${esc(b.doc)}</p><p class="src">Source: ${esc(b.source)}</p></details>
  </article>`;
}
function render(){
  const t = q.value.trim().toLowerCase();
  const hits = DATA.filter(b => (dom === 'all' || b.domain === dom) &&
    (!t || (b.name + ' ' + b.summary + ' ' + b.path + ' ' + b.tags.join(' ')).toLowerCase().includes(t)));
  grid.innerHTML = hits.map(card).join('') || '<p class="note">No block matches. Try a shorter word, or "all" domains.</p>';
  cnt.textContent = hits.length + ' of ' + DATA.length + ' blocks';
}
document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => {
  document.querySelectorAll('.chip').forEach(x => x.setAttribute('aria-pressed', 'false'));
  c.setAttribute('aria-pressed', 'true'); dom = c.dataset.d; render();
  try { localStorage.setItem('lib-dom', dom); } catch (e) {}
}));
q.addEventListener('input', render);
document.addEventListener('click', e => {
  const b = e.target.closest('.copy'); if (!b) return;
  const txt = b.previousElementSibling.textContent;
  const done = () => { b.textContent = 'copied'; setTimeout(() => b.textContent = 'copy', 1200); };
  if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, () => {
    const r = document.createRange(); r.selectNodeContents(b.previousElementSibling);
    const s = getSelection(); s.removeAllRanges(); s.addRange(r); });
});
try { const d = localStorage.getItem('lib-dom'); const c = d && document.querySelector(`.chip[data-d="${d}"]`); if (c) { c.click(); } } catch (e) {}
render();
"""


def build_html():
    E = entries()
    domains = sorted({e["domain"] for e in E})
    chips = '<button class="chip" type="button" data-d="all" aria-pressed="true">all</button>' + "".join(
        f'<button class="chip" type="button" data-d="{d}" aria-pressed="false">{d.replace("_", " ")}</button>' for d in domains)
    els = "".join(
        f'<article class="card el"><div class="top"><div class="fig" aria-hidden="true"><svg viewBox="0 0 160 90">'
        f'<use href="#el-{e["name"]}"/></svg></div><div class="meta"><div class="name">type: {e["name"]}</div>'
        f'<div class="law">{html.escape(e["law"])}</div></div></div><p class="sum">{html.escape(e["what"])}</p>'
        f'<dl class="io"><dt>fields</dt><dd>{html.escape(e["fields"])}</dd><dt>out</dt><dd>force, measure, '
        f'absorbed_work, dissipated{", plastic_set, broken" if e["name"] == "elastoplastic" else ""}'
        f'{", current, battery_current, bus_voltage, soc" if e["name"] == "drive" else ""}</dd></dl>'
        f'<div class="use"><pre>elements:\n{textwrap.indent(html.escape(e["yaml"]), "  ")}</pre>'
        f'<button class="copy" type="button">copy</button></div></article>' for e in ELEMENTS)
    syms = FG.symbols() + "".join(_el_fig(e["name"]) for e in ELEMENTS)
    data = json.dumps(E).replace("</", "<\\/")
    return f"""<title>Simblocks Block Library</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<svg width="0" height="0" style="position:absolute" aria-hidden="true">{syms}</svg>
<div class="wrap">
<div class="eyebrow">simblocks // block library</div>
<h1>Simblocks Block Library</h1>
<p class="lede">Every building block the system-file compiler can use: what it does, what it takes and returns,
the lines to paste into a system file, and a sketch of its geometry with connection points
<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:var(--pt)"/></svg>
and loads <svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true"><line x1="1" y1="5" x2="12" y2="5" style="stroke:var(--acc)" stroke-width="2"/><polygon points="15,5 10,2 10,8" style="fill:var(--acc)"/></svg>.
Generated from the code ({len(E)} blocks), so it always matches the version you have.</p>

<h2>Element types: what connects bodies</h2>
<p class="note">A system file wires bodies together with these {len(ELEMENTS)} element types. Each acts along a
<code>measure</code> (for example <code>hub + $R_tip * rotor - target</code>) and the compiler applies the equal and
opposite force to every body in it. Catalog blocks below supply their numbers (stiffness, yield tables, drag
constants) through <code>derived:</code>.</p>
<div class="els">{els}</div>

<h2>Catalog blocks</h2>
<p class="note">Builders run once before the sim (material plus geometry to numbers). Analytic blocks are
closed-form answers for sizing and checks. Kernels run every time step; the ones marked with an element type
are already driven by the compiler. <code>&lt;name&gt;</code> in a snippet is a value you fill in, with units
(<code>10 mm</code>, <code>12000 rpm</code>).</p>
<div class="bar" role="search"><input id="q" type="search" placeholder="Search blocks: plate, hertz, 7075, windage"
aria-label="Search blocks">{chips}<span class="count" id="count"></span></div>
<div class="grid" id="grid"></div>
</div>
<script>{JS.replace("__DATA__", data)}</script>
"""


def write(path):
    with open(path, "w") as fh:
        fh.write(build_html())
    return path


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else "block_library.html"))

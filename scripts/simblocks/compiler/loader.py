"""YAML loading with line numbers, and the validation rules for system files (format version 1).
Every error names the file, the line and the field path, and says what was expected."""
import re

import yaml


class SpecError(ValueError):
    def __init__(self, msg, path="", line=None, file=""):
        loc = f"{file}:{line}" if line else (file or "")
        super().__init__(f"{loc}{' ' if loc else ''}[{path}] {msg}" if path else f"{loc} {msg}".strip())
        self.path = path; self.line = line


class LDict(dict):
    """dict that remembers the source line of each key (attribute `lines`)."""
    lines = None


class _Loader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    d = LDict()
    d.lines = {}
    for k_node, v_node in node.value:
        k = loader.construct_object(k_node, deep=True)
        if k in d:
            raise SpecError(f"duplicate key '{k}'", line=k_node.start_mark.line + 1)
        d[k] = loader.construct_object(v_node, deep=True)
        d.lines[k] = k_node.start_mark.line + 1
    d.lines["__self__"] = node.start_mark.line + 1
    return d


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load(path_or_text):
    if "\n" not in path_or_text and path_or_text.endswith((".yaml", ".yml")):
        with open(path_or_text) as fh:
            text = fh.read()
        name = path_or_text
    else:
        text, name = path_or_text, "<text>"
    try:
        doc = yaml.load(text, Loader=_Loader)
    except yaml.YAMLError as exc:
        raise SpecError(f"YAML syntax: {exc}", file=name)
    if not isinstance(doc, dict):
        raise SpecError("file must be a mapping (key: value pairs)", file=name)
    doc.lines = getattr(doc, "lines", {}) or {}
    doc["__file__"] = name
    return doc


def line_of(d, key):
    ln = getattr(d, "lines", None) or {}
    return ln.get(key, ln.get("__self__"))


# ---------------------------------------------------------------------------------------------- schema
TOP = {"simblocks", "name", "question", "dim", "units", "params", "derived", "bodies", "elements", "outputs",
       "failure", "uncertain", "limits", "knockouts", "run", "report", "notes", "assumptions", "sources",
       "directions", "angles", "study", "exports", "editor", "gravity", "circuit", "control", "ports", "models", "budget", "scenarios"}
REQUIRED_TOP = {"simblocks", "name", "dim", "bodies", "elements", "outputs", "run"}
BODY_KEYS = {"dof", "mass", "inertia", "shape", "x0", "v0", "ground", "label", "note", "__hidden__"}
BODY_KEYS_3D = {"dof", "axis", "mass", "inertia", "shape", "at", "com", "orient0", "angle0", "v0", "omega0", "points", "spheres",
                "planes", "ground", "label", "note"}
BODY_KEYS_2D = {"dof", "mass", "inertia", "shape", "at", "com", "angle0", "v0", "omega0", "points", "circles", "lines", "ground",
                "label", "note"}
ELEMENT_KEYS = {
    "spring": {"k"}, "dashpot": {"c"}, "kelvin_voigt": {"k", "c"}, "elastoplastic": {"table", "break_at", "strain_rate", "set0", "acc0", "softening"},
    "hertz": {"K", "n", "alpha"}, "friction": {"F_c", "v_eps", "F_s", "v_s"}, "drag": {"K"},
    "force": {"P", "csv", "pulse"}, "drive": {"kv", "R", "I_max", "gear", "cells", "R0", "R1", "C1", "capacity", "throttle",
                              "J_rotor"},
    "dc_motor": {"kv", "Kt", "R", "V", "I_max", "gear", "throttle", "J_rotor"},
    "emotor": {"__part__", "gear", "J_rotor", "__kt__", "__idx__", "__R__"},
    "impact": {"e", "mu", "v_rest"},
    "gear": {"module", "teeth", "face_width", "pressure_angle", "backlash", "k_mesh", "c", "zeta", "material", "E",
             "nu", "internal"},
    "bolt": {"size", "d", "pitch", "grip", "threaded", "preload", "torque", "nut_factor", "grade", "Sy", "E",
             "E_members", "k_bolt", "k_members", "c", "zeta", "Su", "elongation"},
    "pin": {"k", "c", "rigid"}, "contact_friction": {"mu", "v_eps", "contact"},
    "rope": {"k", "c", "length"}, "weld": {"k", "c", "k_rot", "c_rot", "rigid"},
    "hinge": {"k", "c", "k_rot", "c_rot", "axis", "k_twist", "c_twist", "rigid"},
    "slider": {"k", "c", "k_rot", "c_rot", "axis", "friction", "v_eps", "rigid"},
    "beam": {"E", "A", "I", "Iy", "Iz", "J", "G", "EA", "EI", "EIy", "EIz", "GJ", "section", "material", "density",
             "hinge", "damping", "orient", "rotary_inertia", "end_mass"},
    "rocket": {"thrust_vac", "exit_area", "mass_flow", "start", "stop", "rise", "tailoff", "cutoff", "throttle",
               "propellant"},
    "jettison": {"at", "mass"},
    "aero_drag": {"area", "Cd"},
    "steering": {"pitch_start", "pitch_rate", "pitch_end", "tilt_arrest", "guidance_start", "target_altitude",
                 "target_speed", "tgo_freeze"},
    "landing_guidance": {"mode", "start", "stop", "site_range", "target", "freeze", "schedule", "tau", "contact_altitude",
                         "throttle", "end_before", "stop_below"},
}
TYPES_2D_ONLY = {"pin", "contact_friction", "rope", "weld", "hinge", "slider", "beam", "rocket", "jettison", "aero_drag",
                 "steering", "landing_guidance"}
ELEMENT_COMMON = {"type", "measure", "one_sided", "label", "source", "note", "knockout", "mass", "segments",
                  "__lump__", "__hidden__", "__rigid__"}
REQUIRED_ELEMENT = {"spring": {"k"}, "dashpot": {"c"}, "kelvin_voigt": {"k", "c"}, "elastoplastic": {"table"},
                    "hertz": {"K"}, "friction": {"F_c", "v_eps"}, "drag": {"K"}, "force": set(),
                    "drive": {"kv", "R", "I_max", "cells", "R0", "capacity"}, "dc_motor": {"R", "V"}, "emotor": set(), "pin": {"k"},
                    "contact_friction": {"mu", "contact"}, "rope": {"k"}, "weld": {"k", "k_rot"}, "hinge": {"k"},
                    "slider": {"k", "axis"}, "beam": set(), "impact": {"e"},
                    "gear": {"module", "teeth", "face_width"}, "bolt": {"grip"},
                    "rocket": {"thrust_vac", "mass_flow"}, "jettison": {"at", "mass"}, "aero_drag": {"area", "Cd"},
                    "steering": set(), "landing_guidance": {"site_range"}}
OUTPUT_KEYS = {"element", "body", "part", "control", "quantity", "reduce", "unit", "abs", "value", "expr", "label", "note"}
ELEMENT_QUANTITIES = {"force", "measure", "plastic_set", "rate_factor", "absorbed_work", "dissipated", "current",
                      "battery_current", "bus_voltage", "soc", "broken", "moment", "torque", "shear", "hinge_set",
                      "impulse", "friction_impulse", "drift",
                      "pitch", "downrange", "altitude", "speed_inertial", "flight_path_angle", "airspeed",
                      "drag", "dynamic_pressure", "mach", "air_density", "ambient_pressure",
                      "thrust", "mass_flow", "back_pressure", "vehicle_mass", "throttle", "dropped",
                      "thrust_command", "range_to_site", "speed_forward", "speed_up", "propellant",
                      "bending_stress", "contact_stress", "bolt_force", "clamp_force", "bolt_stress", "accumulated_set"}
BODY_QUANTITIES = {"position", "velocity", "kinetic_energy"}
BODY_QUANTITIES_3D = {"x", "y", "z", "vx", "vy", "vz", "speed", "angle", "omega", "wx", "wy", "wz", "position",
                      "velocity", "kinetic_energy"}
BODY_QUANTITIES_2D = {"x", "y", "angle", "vx", "vy", "omega", "speed", "position", "velocity", "kinetic_energy"}
REDUCES = {"final", "max", "min", "max_abs", "first_time_above", "first_time_below", "at", "time_of_max", "time_of_min"}
EXPORT_KEYS = {"name", "columns", "folder", "naming", "time_unit", "points", "header", "when", "label", "note"}
EXPORT_COLUMN_KEYS = {"element", "body", "part", "control", "quantity", "unit", "label"}
HISTORY_QUANTITIES = {"force", "measure"}
STATE_HISTORY = {"elastoplastic": {"plastic_set", "broken", "rate_factor", "accumulated_set"}, "drive": {"current", "battery_current", "bus_voltage", "soc"},
                 "dc_motor": {"current"},
                 "gear": {"bending_stress", "contact_stress"},
                 "bolt": {"bolt_force", "clamp_force", "bolt_stress", "plastic_set", "broken"},
                 "beam": {"moment", "torque", "shear", "hinge_set"},
                 "impact": {"impulse", "dissipated", "friction_impulse"},
                 "weld": {"moment"}, "hinge": {"moment"}, "slider": {"moment"},
                 "steering": {"pitch", "downrange", "altitude", "speed_inertial", "flight_path_angle", "airspeed"},
                 "aero_drag": {"drag", "dynamic_pressure", "mach", "air_density", "ambient_pressure"},
                 "rocket": {"thrust", "mass_flow", "back_pressure", "vehicle_mass", "throttle", "propellant"},
                 "landing_guidance": {"thrust_command", "range_to_site", "altitude", "speed_forward", "speed_up",
                                      "throttle"},
                 "jettison": {"dropped"}}
EXPORT_HEADERS = {"units", "plain", "none"}
EXPORT_ON = {"run", "study"}
FAIL_KEYS = {"name", "metric", "limit", "fails_when", "demand", "capacity", "origin", "assumption", "reach_metric"}


def _unit_ok(u, p, line, f):
    """An output's or export column's unit must be one the program knows, said at load time with its path."""
    if u in (None, ""):
        return
    from .units import unit_factor, UnitError
    try:
        unit_factor(str(u))
    except UnitError as e:
        raise SpecError(f"unknown unit '{u}' ({e})", p, line, f) from None


_SECTION_KIND = {"params": dict, "derived": dict, "bodies": dict, "elements": dict, "outputs": dict, "run": dict,
                 "circuit": dict, "control": dict, "uncertain": dict, "study": dict, "ports": dict, "models": dict,
                 "budget": dict, "scenarios": dict, "knockouts": dict, "editor": dict, "failure": list,
                 "exports": list, "limits": list}


def check(doc):
    f = doc.get("__file__", "")
    if not isinstance(doc, dict):
        raise SpecError("a system file is a YAML mapping (key: value lines)", "", None, f)
    unknown = set(doc) - TOP - {"__file__"}
    for k in unknown:
        raise SpecError(f"unknown top-level key '{k}' (allowed: {', '.join(sorted(TOP))})", k, line_of(doc, k), f)
    for k, kind in _SECTION_KIND.items():         # a section with the wrong shape: say so here, not as a crash later
        v = doc.get(k)
        if k in doc and v is not None and not isinstance(v, kind):
            raise SpecError(f"'{k}' must be a {'mapping (name: value lines)' if kind is dict else 'list (- item lines)'}",
                            k, line_of(doc, k), f)
        if isinstance(v, dict):                    # YAML reads on / off / yes / no / 1 as true, false or a number
            for kk in v:
                if not isinstance(kk, str):
                    raise SpecError(f"the name {kk!r} under '{k}' is read by YAML as {type(kk).__name__}: quote it "
                                    "('on') or pick another name", k, line_of(doc, k), f)
        if k in doc and v is None and k in ("bodies", "elements", "outputs", "run"):
            doc[k] = {} if k != "run" else None
    if doc.get("run") is None and "run" in doc:
        raise SpecError("run needs t_end (run: {t_end: 1 s})", "run", line_of(doc, "run"), f)
    if doc.get("models"):                        # a sequence of model blocks needs no mechanics of its own
        doc.setdefault("dim", "1d"); doc.setdefault("outputs", {}); doc.setdefault("elements", {})
        if not doc.get("bodies"):
            doc["bodies"] = {"__ground__": {"ground": True}}
        doc.setdefault("run", {"t_end": 1e-6})
    if isinstance(doc.get("budget"), dict) and doc["budget"].get("items") is not None \
            and not isinstance(doc["budget"]["items"], dict):
        raise SpecError("budget.items is a mapping: one part per line (wiring: 25 g)", "budget.items",
                        line_of(doc["budget"], "items"), f)
    if isinstance(doc.get("budget"), dict) and "limit" in doc["budget"]:   # weight budget: two outputs of its own
        lim = doc["budget"]["limit"]
        u = str(lim).strip().split(None, 1)[1] if isinstance(lim, str) and len(str(lim).split()) > 1 else "kg"
        outs_ = doc.get("outputs")
        if outs_ is None:
            outs_ = doc["outputs"] = {}
        outs_.setdefault("mass_total", {"expr": "$__mass_total", "unit": u, "label": "weight total"})
        outs_.setdefault("mass_margin", {"expr": "$__mass_margin", "unit": u, "label": "weight margin (limit - total)"})
    if "elements" not in doc and doc.get("circuit"):
        doc["elements"] = {}                     # a circuit-only model: the motors are the elements
    if not doc.get("bodies") and doc.get("circuit"):
        doc["bodies"] = {"__ground__": {"ground": True}}      # a pure circuit: no mechanics to move
    for k in REQUIRED_TOP:
        if k not in doc:
            raise SpecError(f"missing required key '{k}'", k, None, f)
    if doc["simblocks"] != 1:
        raise SpecError("this compiler reads format version 1 (simblocks: 1)", "simblocks", line_of(doc, "simblocks"), f)
    if doc["dim"] not in ("1d", "2d", "3d"):
        raise SpecError("dim must be 1d, 2d or 3d", "dim", line_of(doc, "dim"), f)
    three = doc["dim"] == "3d"
    two = doc["dim"] in ("2d", "3d")          # 2d and 3d share the geometric element rules
    for k in ("directions", "angles"):
        if k in doc:
            raise SpecError(f"'{k}' is reserved and not used: in 2d give directions with along: dir and "
                            "contact geometry with points, circles and lines on the bodies", k, line_of(doc, k), f)
    if "gravity" in doc and not two:
        raise SpecError("gravity is a 2d / 3d setting; in 1d add a force element (P = m g) on the body", "gravity",
                        line_of(doc, "gravity"), f)
    bodies = doc["bodies"]
    if not isinstance(bodies, dict) or not bodies:
        raise SpecError("bodies must be a non-empty mapping", "bodies", line_of(doc, "bodies"), f)
    for name, b in bodies.items():
        p = f"bodies.{name}"
        if not isinstance(b, dict):
            raise SpecError("each body is a mapping", p, line_of(bodies, name), f)
        if three:
            for k in set(b) - BODY_KEYS_3D:
                raise SpecError(f"unknown 3d body key '{k}' (allowed: {', '.join(sorted(BODY_KEYS_3D))})", p,
                                line_of(b, k), f)
            if b.get("ground"):
                continue
            dof = b.get("dof", "free")
            dl = {"free": ["x", "y", "z", "rotate"], "translation": ["x", "y", "z"], "spin": ["spin"],
                  "rotation": ["spin"]}.get(dof, [dof]) if isinstance(dof, str) else dof
            if not isinstance(dl, list) or not dl or any(c not in ("x", "y", "z", "spin", "rotate") for c in dl):
                raise SpecError("3d dof is a list of x, y, z plus spin (about axis:) or rotate, or 'free'", p,
                                line_of(b, "dof") or line_of(bodies, name), f)
            if any(c in dl for c in ("x", "y", "z")) and "mass" not in b and "shape" not in b:
                raise SpecError("a body free to translate needs 'mass'", p, line_of(bodies, name), f)
            if any(c in dl for c in ("spin", "rotate")) and "inertia" not in b and "shape" not in b:
                raise SpecError("a body that turns needs 'inertia' (about its centre of mass)", p, line_of(bodies, name), f)
            continue
        if two:
            for k in set(b) - BODY_KEYS_2D:
                hint = {"x0": " (2d uses at: [x, y])", "position": " (2d uses at: [x, y])"}.get(k, "")
                raise SpecError(f"unknown 2d body key '{k}'{hint} (allowed: {', '.join(sorted(BODY_KEYS_2D))})", p,
                                line_of(b, k), f)
            if b.get("ground"):
                continue
            dof = b.get("dof", "planar")
            dl = {"planar": ["x", "y", "rz"], "translation": ["x", "y"], "rotation": ["rz"], "spin": ["rz"]}.get(dof, dof) \
                if isinstance(dof, str) else dof
            if not isinstance(dl, list) or not dl or any(c not in ("x", "y", "rz") for c in dl):
                raise SpecError("2d dof is a list of free coordinates, any of x, y, rz (or planar, translation, "
                                "rotation)", p, line_of(b, "dof") or line_of(bodies, name), f)
            if ("x" in dl or "y" in dl) and "mass" not in b and "shape" not in b:
                raise SpecError("a body free in x or y needs 'mass'", p, line_of(bodies, name), f)
            if "rz" in dl and "inertia" not in b and "shape" not in b:
                raise SpecError("a body free to spin (rz) needs 'inertia' (about its centre of mass)", p,
                                line_of(bodies, name), f)
            continue
        for k in set(b) - BODY_KEYS:
            raise SpecError(f"unknown body key '{k}' (allowed: {', '.join(sorted(BODY_KEYS))})", p, line_of(b, k), f)
        if b.get("ground"):
            continue
        dof = b.get("dof")
        if dof not in ("translation", "rotation"):
            raise SpecError("dof must be 'translation' or 'rotation'", p, line_of(b, "dof") or line_of(bodies, name), f)
        need = "mass" if dof == "translation" else "inertia"
        wrong = "inertia" if dof == "translation" else "mass"
        if wrong in b:
            raise SpecError(f"a {dof} body takes '{need}', not '{wrong}'", p, line_of(b, wrong), f)
        if need not in b and "shape" not in b:
            raise SpecError(f"a {dof} body needs '{need}'", p, line_of(bodies, name), f)
    els = doc["elements"]
    has_circ = (isinstance(doc.get("circuit"), dict) and bool((doc["circuit"].get("parts") or {}))) or bool(doc.get("models"))
    if els is None and has_circ:
        els = doc["elements"] = {}
    if not isinstance(els, dict) or (not els and not has_circ):
        raise SpecError("elements must be a non-empty mapping (or the model needs a circuit)", "elements",
                        line_of(doc, "elements"), f)
    connected = set()
    for name, e in els.items():
        p = f"elements.{name}"
        if not isinstance(e, dict) or "type" not in e:
            raise SpecError("each element needs a 'type'", p, line_of(els, name), f)
        t = e["type"]
        if t in TYPES_2D_ONLY and not two:
            raise SpecError(f"{t} is a 2d / 3d element (it needs geometry)", p, line_of(e, "type"), f)
        if t not in ELEMENT_KEYS:
            raise SpecError(f"unknown element type '{t}' (allowed: {', '.join(sorted(ELEMENT_KEYS))})", p,
                            line_of(e, "type"), f)
        for k in set(e) - ELEMENT_KEYS[t] - ELEMENT_COMMON:
            raise SpecError(f"'{k}' is not a field of a {t} element (allowed: "
                            f"{', '.join(sorted(ELEMENT_KEYS[t] | ELEMENT_COMMON))})", p, line_of(e, k), f)
        if two and "mass" in e and "mass" not in ELEMENT_KEYS[t] and t != "beam" and "__" not in str(name):
            # a 1D element lumps its own mass onto its bodies; a 2D / 3D element would need where along it the mass
            # sits, so it is refused rather than ignored
            raise SpecError("in 2D and 3D an element carries no mass of its own: add it to the bodies it joins (about "
                            "half to each end) and list it in the weight budget's items if you track weight", p,
                            line_of(e, "mass"), f)
        req = REQUIRED_ELEMENT[t] - ({"k", "k_rot"} if e.get("rigid") is True else set())
        if "rigid" in e and not isinstance(e["rigid"], bool):
            raise SpecError("rigid is true or false", p, line_of(e, "rigid"), f)
        for k in req - set(e):
            raise SpecError(f"a {t} element needs '{k}'", p, line_of(els, name), f)
        if t == "dc_motor" and ("kv" in e) == ("Kt" in e):
            raise SpecError("a dc_motor needs kv (speed per volt, e.g. 1350 rpm/V) or Kt (torque per amp, N m/A), "
                            "one of them", p, line_of(els, name), f)
        if t == "contact_friction":
            if "measure" in e:
                raise SpecError("contact_friction takes contact: <contact element> instead of a measure", p,
                                line_of(e, "measure"), f)
            continue
        if "measure" not in e:
            raise SpecError("every element needs a 'measure' (e.g. 'mass - frame')", p, line_of(els, name), f)
        if two:
            import re as _re
            for tok in _re.findall(r"(?<![$\w.])([A-Za-z_]\w*)(?=\.|\b)", str(e["measure"])):
                if tok in bodies:
                    connected.add(tok)
            if e.get("one_sided") and t not in ("kelvin_voigt", "elastoplastic"):
                raise SpecError(f"one_sided applies to kelvin_voigt and elastoplastic only", p, line_of(e, "one_sided"), f)
            continue
        if e.get("one_sided") and t not in ("kelvin_voigt", "elastoplastic"):
            raise SpecError(f"one_sided applies to kelvin_voigt and elastoplastic only ({t} is "
                            f"{'always compression-only' if t == 'hertz' else 'two-sided'})", p, line_of(e, "one_sided"), f)
        for tok in re.findall(r"(?<![$\w.])([A-Za-z_]\w*)\b", str(e["measure"])):     # parentheses and all
            if tok in bodies:
                connected.add(tok)
    circ_parts = ((doc.get("circuit") or {}).get("parts") or {}) if isinstance(doc.get("circuit"), dict) else {}
    for pd in circ_parts.values():                 # a circuit motor turns the body in its measure
        if isinstance(pd, dict) and "measure" in pd:
            for tok in str(pd["measure"]).replace("-", " ").replace("+", " ").replace("*", " ").replace(".", " ") \
                    .replace("[", " ").replace("]", " ").replace(",", " ").replace("{", " ").replace("}", " ").split():
                if tok in bodies:
                    connected.add(tok)
    for name, b in bodies.items():
        if two and doc.get("gravity") not in (None, 0, "0", "none", "off"):
            break                                  # with gravity a body may fly free
        if not b.get("ground") and name not in connected:
            raise SpecError(f"body '{name}' is not in any element's measure: it would never move or load anything",
                            f"bodies.{name}", line_of(bodies, name), f)
    outs = doc["outputs"]
    for name, o in outs.items():
        p = f"outputs.{name}"
        if not isinstance(o, dict):
            raise SpecError("each output is a mapping", p, line_of(outs, name), f)
        for k in set(o) - OUTPUT_KEYS:
            raise SpecError(f"unknown output key '{k}'", p, line_of(o, k), f)
        _unit_ok(o.get("unit"), p, line_of(o, "unit"), f)
        if str(o.get("unit", "")).strip() == "Hz" and o.get("quantity") in ("omega", "wx", "wy", "wz", "velocity", "speed"):
            raise SpecError("Hz is a frequency (1/s): a spin rate in revolutions per second is rps (or rpm, or rad/s)", p,
                            line_of(o, "unit"), f)
        if "expr" in o:
            continue
        if sum(k in o for k in ("element", "body", "part", "control")) != 1:
            raise SpecError("an output names exactly one 'element', 'body', circuit 'part' or 'control' signal (or "
                            "gives an 'expr')", p, line_of(outs, name), f)
        if "control" in o:
            if o["control"] not in (doc.get("control") or {}):
                raise SpecError(f"no control signal named '{o['control']}'", p, line_of(o, "control"), f)
            red = o.get("reduce", "final")
            if red not in REDUCES:
                raise SpecError(f"reduce must be one of {', '.join(sorted(REDUCES))}", p, line_of(o, "reduce"), f)
            if red in ("at",) and "value" not in o or red.startswith("first_time") and "value" not in o:
                raise SpecError(f"{red} needs a 'value'", p, line_of(o, "reduce"), f)
            continue
        if "part" in o:
            from .circuit import QUANTITIES as CQ
            pn_ = str(o["part"])                   # a battery's or a supply's I_load runs as its own part, <name>__load
            if pn_ not in circ_parts and not (pn_.endswith("__load") and isinstance(circ_parts.get(pn_[:-6]), dict)
                                              and circ_parts[pn_[:-6]].get("I_load") not in (None, "")):
                raise SpecError(f"no circuit part named '{o['part']}'", p, line_of(o, "part"), f)
            if o.get("quantity") not in CQ:
                raise SpecError(f"a part's quantity is one of {', '.join(sorted(CQ))}", p, line_of(o, "quantity"), f)
            red = o.get("reduce", "final")
            if red not in REDUCES or red.startswith("first_time"):
                raise SpecError("a part's reduce is final, max, min or max_abs", p, line_of(o, "reduce"), f)
            continue
        if "element" in o and o["element"] not in els:
            raise SpecError(f"no element named '{o['element']}'", p, line_of(o, "element"), f)
        if "body" in o and o["body"] not in bodies:
            raise SpecError(f"no body named '{o['body']}'", p, line_of(o, "body"), f)
        qset = ELEMENT_QUANTITIES if "element" in o else (BODY_QUANTITIES_3D if three else (BODY_QUANTITIES_2D if two else BODY_QUANTITIES))
        if o.get("quantity") not in qset:
            raise SpecError(f"quantity must be one of {', '.join(sorted(qset))}", p, line_of(o, "quantity"), f)
        red = o.get("reduce", "final")
        if red not in REDUCES:
            raise SpecError(f"reduce must be one of {', '.join(sorted(REDUCES))}", p, line_of(o, "reduce"), f)
        if red == "at" and "value" not in o:
            raise SpecError("reduce: at needs a 'value': the time to read the quantity at", p, line_of(o, "reduce"), f)
        if red.startswith("first_time") and "value" not in o:
            raise SpecError(f"{red} needs a 'value' threshold", p, line_of(o, "reduce"), f)
    for i, m in enumerate(doc.get("failure", []) or []):
        p = f"failure[{i}]"
        for k in set(m) - FAIL_KEYS:
            raise SpecError(f"unknown failure key '{k}'", p, line_of(m, k), f)
        if "name" not in m or "origin" not in m:
            raise SpecError("every failure mode needs a 'name' and an 'origin' (who defined it, in their words)",
                            p, line_of(m, "__self__"), f)
        has_metric = "metric" in m and "limit" in m
        has_dc = "demand" in m and "capacity" in m
        if has_metric == has_dc:
            raise SpecError("give either metric + limit, or demand + capacity", p, line_of(m, "__self__"), f)
        if has_metric and m["metric"] not in outs:
            raise SpecError(f"metric '{m['metric']}' is not an output", p, line_of(m, "metric"), f)
    check_exports(doc)
    from .chain import check_ports
    check_ports(doc)
    run = doc["run"]
    if "t_end" not in run:
        raise SpecError("run needs t_end", "run", line_of(doc, "run"), f)
    return True


def check_exports(doc):
    """exports: CSV time histories written next to the model (or any folder) for spreadsheets and other analysis tools."""
    f = doc.get("__file__", "")
    ex = doc.get("exports")
    if ex is None:
        return
    if not isinstance(ex, list):
        raise SpecError("exports must be a list of CSV exports", "exports", line_of(doc, "exports"), f)
    els, bods, seen = doc.get("elements") or {}, doc.get("bodies") or {}, set()
    for i, x in enumerate(ex):
        p = f"exports[{i}]"
        if not isinstance(x, dict):
            raise SpecError("each export is a mapping with name and columns", p, None, f)
        for k in set(x) - EXPORT_KEYS:
            raise SpecError(f"unknown export key '{k}' (allowed: {', '.join(sorted(EXPORT_KEYS))})", p, line_of(x, k), f)
        if not x.get("name"):
            raise SpecError("every export needs a name", p, line_of(x, "__self__"), f)
        if x["name"] in seen:
            raise SpecError(f"two exports are both called '{x['name']}'", p, line_of(x, "name"), f)
        seen.add(x["name"])
        cols = x.get("columns") or []
        if not cols:
            raise SpecError("an export needs at least one column", p, line_of(x, "__self__"), f)
        if str(x.get("header", "units")) not in EXPORT_HEADERS:
            raise SpecError(f"header must be one of {', '.join(sorted(EXPORT_HEADERS))}", p, line_of(x, "header"), f)
        if str(x.get("when", "run")) not in EXPORT_ON:
            raise SpecError("when must be run (the nominal run) or study (the run and every study run)", p, line_of(x, "when"), f)
        try:
            if str(x.get("points", 1000)) != "all" and int(x.get("points", 1000)) < 2:
                raise ValueError
        except (TypeError, ValueError):
            raise SpecError("points must be a whole number of rows (2 or more) or 'all' (every time step)", p, line_of(x, "points"), f)
        _unit_ok(x.get("time_unit"), p, line_of(x, "time_unit"), f)
        for j, c in enumerate(cols):
            q = f"{p}.columns[{j}]"
            if not isinstance(c, dict):
                raise SpecError("a column is {element: name, quantity: force} or {body: name, quantity: velocity}", q, None, f)
            for k in set(c) - EXPORT_COLUMN_KEYS:
                raise SpecError(f"unknown column key '{k}'", q, line_of(c, k), f)
            _unit_ok(c.get("unit"), q, line_of(c, "unit"), f)
            if sum(k in c for k in ("element", "body", "part", "control")) != 1:
                raise SpecError("a column names one element, body, circuit part or control signal", q, line_of(c, "__self__"), f)
            if "control" in c:
                if c["control"] not in (doc.get("control") or {}):
                    raise SpecError(f"no control signal named '{c['control']}'", q, line_of(c, "control"), f)
            elif "part" in c:
                from .circuit import QUANTITIES as CQ
                if c["part"] not in ((doc.get("circuit") or {}).get("parts") or {}):
                    raise SpecError(f"no circuit part named '{c['part']}'", q, line_of(c, "part"), f)
                if c.get("quantity", "current") not in CQ:
                    raise SpecError(f"a part column exports one of: {', '.join(sorted(CQ))}", q, line_of(c, "quantity"), f)
            elif "element" in c:
                e = els.get(c["element"])
                if e is None:
                    raise SpecError(f"element '{c['element']}' is not defined", q, line_of(c, "element"), f)
                qn = c.get("quantity", "force")
                ok = HISTORY_QUANTITIES | STATE_HISTORY.get(e.get("type"), set())
                if qn not in ok:
                    raise SpecError(f"'{qn}' has no time history for a {e.get('type')} element "
                                    f"(export one of: {', '.join(sorted(ok))})", q, line_of(c, "quantity"), f)
            else:
                if c["body"] not in bods:
                    raise SpecError(f"body '{c['body']}' is not defined", q, line_of(c, "body"), f)
                if bods[c["body"]].get("ground"):
                    raise SpecError(f"'{c['body']}' is ground and never moves", q, line_of(c, "body"), f)
                okq = ("x", "y", "angle", "vx", "vy", "omega", "speed", "position", "velocity") if doc.get("dim") == "2d" \
                    else (tuple(sorted(BODY_QUANTITIES_3D - {"kinetic_energy"})) if doc.get("dim") == "3d" else ("position", "velocity"))
                if c.get("quantity", "position") not in okq:
                    raise SpecError(f"a body column exports one of: {', '.join(okq)}", q, line_of(c, "quantity"), f)


def locate(doc, path):
    """Best-effort source line for a dotted path like 'elements.shaft.table' or 'failure[0]'."""
    import re
    node, line = doc, None
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            i = int(part[1:-1])
            if isinstance(node, list) and i < len(node):
                node = node[i]
                line = line_of(node, "__self__") if isinstance(node, dict) else line
            else:
                break
        elif isinstance(node, dict) and part in node:
            line = line_of(node, part) or line
            node = node[part]
        else:
            break
    return line


def with_location(exc, doc):
    if exc.line is None and exc.path:
        ln = locate(doc, exc.path)
        msg = str(exc).split("] ", 1)[-1] if "] " in str(exc) else str(exc)
        return SpecError(msg, exc.path, ln, doc.get("__file__", ""))
    return exc

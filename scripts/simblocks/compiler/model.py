"""Compile a checked system file into engine arrays, run it, and reduce the outputs.

compile_model(doc, overrides) -> Compiled   (pure function of the file + parameter overrides)
Compiled.run(dt=None, history=False) -> dict of outputs (+ history)
"""
import inspect
import math
import re

import numpy as np

from .expr import eval_numeric, eval_measure, ExprError
from .loader import SpecError, line_of
from .units import parse_quantity, unit_factor, UnitError
from . import engine1d as E

TYPE_CODE = {"spring": 0, "dashpot": 1, "kelvin_voigt": 2, "elastoplastic": 3, "hertz": 4, "friction": 5,
             "drag": 6, "force": 7, "drive": 8, "impact": 14, "emotor": 17}
AUX = {"plastic_set": 0, "broken": 2, "current": 0, "battery_current": 1, "bus_voltage": 2, "soc": 3,
       "moment": 0, "torque": 1, "shear": 2, "hinge_set": 3, "impulse": 0, "friction_impulse": 2, "drift": 3,
       # vehicles in flight: steering (altitude, inertial speed, flight path angle, speed through the air),
       # aero_drag (dynamic pressure, Mach, density, outside pressure), rocket (flow, back pressure, vehicle mass)
       "altitude": 0, "speed_inertial": 1, "flight_path_angle": 2, "airspeed": 3,
       "dynamic_pressure": 0, "mach": 1, "air_density": 2, "ambient_pressure": 3,
       "mass_flow": 0, "back_pressure": 1, "vehicle_mass": 2, "throttle": 3,
       # landing guidance: altitude (0), forward speed over the ground, climb rate, throttle level (3)
       "speed_forward": 1, "speed_up": 2}
QALIAS = {"pitch": "force", "thrust": "force", "drag": "force", "dropped": "force", "downrange": "measure",
          "thrust_command": "force", "range_to_site": "measure", "propellant": "measure"}
KNOCK = "__knock__"          # override prefix: element switched off
ELASTIC = "__elastic__"      # override prefix: elastoplastic element made elastic


class Text(str):
    """A parameter that is text (material name, layup), not a number."""
    why = ""


# ------------------------------------------------------------------------------------------- values
def resolve(v, ctx, where):
    """Number / 'n unit' / '$ref' / '= expr' / {block: ...} / list / plain text -> python value."""
    try:
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list):
            return [resolve(x, ctx, f"{where}[{i}]") for i, x in enumerate(v)]
        if isinstance(v, dict):
            if "block" in v:
                return call_block(v, ctx, where)
            return {k: resolve(x, ctx, f"{where}.{k}") for k, x in v.items() if k != "lines"}
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("$") and all(p_.isidentifier() for p_ in s[1:].split(".")):
                head, *rest = s[1:].split(".")
                if head not in ctx:
                    raise SpecError(f"undefined parameter '${head}'", where)
                val = ctx[head]
                for r_ in rest:                       # $plate.k -> field of a block result
                    if isinstance(val, dict) and r_ in val:
                        val = val[r_]
                    else:
                        raise SpecError(f"'${head}' has no field '{r_}'"
                                        + (f" (fields: {', '.join(val)})" if isinstance(val, dict) else ""), where)
                return float(val) if isinstance(val, (int, np.integer)) and not isinstance(val, bool) else val
            if s.startswith("="):
                num = numeric_namespace(ctx)
                try:
                    return float(eval_numeric(s, num))
                except ExprError as exc:
                    txt = [k for k in ctx if isinstance(ctx[k], str) and f"${k}" in s]
                    hint = f" ('{txt[0]}' is text, not a number)" if txt else ""
                    raise SpecError(f"{exc}{hint}", where)
            try:
                return parse_quantity(s)
            except UnitError as exc:
                t = Text(s)
                t.why = str(exc)          # kept as text (material name, layup); an error if later used as a number
                return t
        raise SpecError(f"cannot read value {v!r}", where)
    except (UnitError, ExprError) as exc:
        raise SpecError(str(exc), where)


_CACHE = {}       # builder results keyed by (block, args): Monte Carlo reruns reuse unchanged builds


_MAT_PROP = re.compile(r"^(E|nu|sy|Sy|su|Su|Y|rho|density|G)(\d*|_\w*)$")
_PROP_ATTR = {"E": "E", "nu": "nu", "sy": "sy", "Sy": "sy", "Y": "sy", "su": "su", "Su": "su", "rho": "rho",
              "density": "rho", "G": "G"}


def _material_prop(text, k, block, where):
    """A material name given where a block wants one of its properties: '4140 prehard' into E1 gives its E."""
    from ..materials import get
    from ..materials.curves import resolve_material
    try:
        m = get(text)
    except Exception:
        try:
            m = get(resolve_material(text))
        except Exception:
            raise SpecError(f"block '{block}' argument '{k}' needs a number; '{text}' is not a number or a known "
                            "material", where)
    base = _MAT_PROP.match(k).group(1)
    attr = _PROP_ATTR[base]
    v = getattr(m, attr, None)
    if attr == "G" and not v and getattr(m, "E", None):
        nu = getattr(m, "nu", None) or 0.3
        v = m.E / (2 * (1 + nu))
    if v is None or (isinstance(v, float) and v != v):
        raise SpecError(f"material '{text}' has no {attr} for block '{block}' argument '{k}'", where)
    return float(v)


def call_block(d, ctx, where):
    from ..registry import REGISTRY
    from ..materials import get
    from ..materials.curves import resolve_material
    name = d["block"]
    if name not in REGISTRY:
        raise SpecError(f"no block named '{name}' (see references/CATALOG.md or `simblocks catalog`)", where)
    fn = REGISTRY[name].func
    py = getattr(fn, "py_func", fn)
    sig = inspect.signature(py)
    args = {}
    for k, x in d.items():
        if k in ("block", "take", "lines"):
            continue
        if k not in sig.parameters:
            raise SpecError(f"block '{name}' has no argument '{k}' (it takes: {', '.join(sig.parameters)})", where)
        val = resolve(x, ctx, f"{where}.{k}")
        if k == "mat" and isinstance(val, str):
            val = get(resolve_material(val))
        elif isinstance(val, str) and _MAT_PROP.match(k) and not (k == "G" and "gear" in (py.__doc__ or "").lower()):
            # (a block whose G is a gear ratio never takes a material there)
            # a material plugged straight into a property argument (E1, nu_t, sy, Y_o, rho...): take that property
            val = _material_prop(val, k, name, where)
        args[k] = val
    missing = [p for p, prm in sig.parameters.items() if prm.default is inspect._empty and p not in args]
    if missing:
        raise SpecError(f"block '{name}' needs argument(s): {', '.join(missing)}", where)
    key = (name, repr(sorted(args.items(), key=lambda kv: kv[0])))
    if key in _CACHE:
        out = _CACHE[key]
    else:
        out = None
    try:
        if out is None:
            out = fn(**args)
            if len(_CACHE) > 512:
                _CACHE.clear()
            _CACHE[key] = out
    except SpecError:
        raise
    except Exception as exc:
        raise SpecError(f"block '{name}' failed: {exc}", where)
    take = d.get("take")
    if take is None:
        return out
    try:
        return out[take]
    except Exception:
        keys = list(out.keys()) if isinstance(out, dict) else f"0..{len(out) - 1}"
        raise SpecError(f"block '{name}' output has no '{take}' (has: {keys})", where)


def numeric_namespace(ctx):
    """Numbers usable in '=' expressions: numeric params plus numeric fields of block results ($plate.k)."""
    out = {}
    for k, x in ctx.items():
        if isinstance(x, (int, float, np.floating)) and not isinstance(x, bool):
            out[k] = float(x)
        elif isinstance(x, dict):
            for f_, y in x.items():
                if isinstance(y, (int, float, np.floating)) and not isinstance(y, bool):
                    out[f"{k}.{f_}"] = float(y)
    return out


def num(v, where):
    if isinstance(v, Text):
        raise SpecError(f"expected a number, got text '{v}'" + (f" (as a quantity: {v.why})" if v.why else ""), where)
    if isinstance(v, bool) or not isinstance(v, (int, float, np.floating)):
        raise SpecError(f"expected a number here, got {v!r}", where)
    return float(v)


# ------------------------------------------------------------------------------------------- compile
class Compiled:
    pass


def build_context(doc, overrides=None):
    f = doc.get("__file__", "")
    ctx = {}
    params = doc.get("params", {}) or {}
    ov = dict(overrides or {})
    for k, v in params.items():
        where = f"params.{k}"
        orig = v.get("value") if isinstance(v, dict) and "value" in v else v
        if k in ov:
            o_ = ov.pop(k)
            if isinstance(resolve(orig, ctx, where), str) and not isinstance(o_, (dict, list)):
                ctx[k] = Text(str(o_)); continue          # text param (material, layup) stays text
            orig = o_
        val = resolve(orig, ctx, where)
        ctx[k] = val
    for k in [k for k in ov if not k.startswith("__")]:
        raise SpecError(f"override '{k}' is not a parameter of this file", "params", None, f)
    for k, v in _derived_order(doc.get("derived", {}) or {}):
        try:
            ctx[k] = resolve(v, ctx, f"derived.{k}")
        except SpecError as exc:
            exc.args = (f"{f}:{line_of(doc['derived'], k)} {exc.args[0]}",)
            raise
    return ctx


def _derived_order(der):
    """Derived values in an order where each comes after the ones it uses ($name or $name.field), keeping the
    file's order otherwise (the editor may write them grouped by kind)."""
    import re as _re
    names = list(der)
    deps = {}
    for k, v in der.items():
        txt = str(v)
        deps[k] = {m for m in _re.findall(r"\$([A-Za-z_]\w*)", txt) if m in der and m != k}
    out, done, busy = [], set(), set()

    def visit(k):
        if k in done or k in busy:            # a cycle is left to the resolver to report
            return
        busy.add(k)
        for d in sorted(deps[k], key=names.index):
            visit(d)
        busy.discard(k); done.add(k); out.append(k)
    for k in names:
        visit(k)
    return [(k, der[k]) for k in out]


CTL_D_TYPES = {0, 2, 3, 4, 9, 12}            # spring-like: a controlled one has no stored-energy law of its own


def compile_model(doc, overrides=None):
    """Compile a system file. Fields written {control: signal} are found by compiling once with marker values (which
    parameter slot each one lands in, as itself or as its inverse, like a motor's kv), then the model compiles with
    their first-step values. A field with max: is also compiled at its max, and the time step is sized for that."""
    from .control import controlled_fields, with_values, compile_control, SENTINEL, empty_control
    fields = controlled_fields(doc)
    if not doc.get("control") and not fields:
        return _compile_model(doc, overrides)
    sec = doc.get("control") or {}
    fname = doc.get("__file__", "")
    acts = []
    if fields:
        marks = [SENTINEL * (i + 1) for i in range(len(fields))]
        Ms = _compile_model(with_values(doc, fields, marks), overrides)
        ctx0 = Ms.ctx
        cp = Ms.circ["arrays"]["prm"] if getattr(Ms, "circ", None) else np.zeros((0, 1))
        PP = Ms.arrays["P"]
        for i, fd in enumerate(fields):
            where = fd["where"]
            if fd["signal"] not in sec:
                raise SpecError(f"no control signal named '{fd['signal']}' (signals: {', '.join(sec) or 'none'})", where,
                                None, fname)
            m = marks[i]
            found = []                                   # (array code, row, slot): 0 P, 1 circuit; +2 when inverted
            for code, arr in ((0, PP), (1, cp)):
                for inv, val in ((0, m), (2, 1.0 / m)):
                    r_, c_ = np.where(np.isclose(arr, val, rtol=1e-9, atol=0.0)) if arr.size else ((), ())
                    found += [(code + inv, int(r), int(c)) for r, c in zip(r_, c_)]
            if fd["path"][0] == "elements":
                en = fd["path"][1]
                if en not in Ms.el_names:
                    raise SpecError("a split element (segments) cannot be driven by a control signal", where, None, fname)
                found = [x for x in found if x[0] in (0, 2) and x[1] == Ms.el_names.index(en)]
            elif any(x[0] in (0, 2) for x in found):
                found = []                               # also copied into the mechanics (a motor's Kt): not one number
            if len(found) != 1:
                raise SpecError(f"'{fd['path'][-1]}' cannot be driven by a control signal (the model turns it into "
                                "other numbers before it runs)", where, None, fname)
            acts.append((found[0][1], found[0][2], list(sec).index(fd["signal"]), found[0][0], i))
        num_ = lambda v, w: num(resolve(v, ctx0, w), w)
        inits = [num_(fd["init"], fd["where"] + ".init") for fd in fields]
        maxes = [num_(fd["max"], fd["where"] + ".max") if fd["max"] is not None else None for fd in fields]
        M = _compile_model(with_values(doc, fields, inits), overrides)
        if any(x is not None for x in maxes):
            # the time step: a driven stiffness or damping is stable at its first-step value, but must stay stable at
            # the most it reaches. Size the step (auto, or held in run bounds) for the declared max
            Mx = _compile_model(with_values(doc, fields, [x if x is not None else v for x, v in zip(maxes, inits)]),
                                overrides)
            if Mx.dt_crit < M.dt_crit:
                M.dt_crit = Mx.dt_crit
                M.dt_auto = min(M.dt_auto, Mx.dt_auto)
                if (doc.get("run") or {}).get("dt", "auto") == "auto":
                    M.dt = min(M.dt, Mx.dt)
        # fields with no max that move the stability limit: flagged after the run with the most they reached
        probes = []
        for i, fd in enumerate(fields):
            if maxes[i] is None:
                v = [inits[j] for j in range(len(fields))]
                v[i] = 4.0 * inits[i] if inits[i] else 1.0
                try:
                    Mp = _compile_model(with_values(doc, fields, v), overrides)
                    if Mp.dt_crit < 0.999 * M.dt_crit:
                        probes.append(i)
                except SpecError:
                    pass
        M.ctl_fields = [dict(fd, init_v=inits[i], max_v=maxes[i], sensitive=i in probes) for i, fd in enumerate(fields)]
        if getattr(M, "adaptive", False):
            fits = [_field_fit(doc, overrides, fields, inits, maxes, i, M) for i in range(len(fields))]
            M.dt_fits = np.array([fits[a[4]] for a in acts], float).reshape(len(acts), 4)
    else:
        M = _compile_model(doc, overrides)
        M.ctl_fields = []
    c = compile_control(M.doc, M.ctx, M, lambda n, q, w: sensor_slot(M, n, q, w))
    base = empty_control()
    if c is None:
        c = dict(names=[], sensors=[], **{k: v for k, v in base.items() if k not in ("ae", "aslot", "asig", "aarr", "ctlD")})
    ne = len(M.el_names)
    ctlD = np.zeros(max(ne, 1), np.int64)
    for e, slot, sg, arr, _i in acts:
        if arr in (0, 2) and int(M.arrays["typ"][e]) in CTL_D_TYPES:
            ctlD[e] = 1
    c.update(ae=np.array([a[0] for a in acts], np.int64), aslot=np.array([a[1] for a in acts], np.int64),
             asig=np.array([a[2] for a in acts], np.int64), aarr=np.array([a[3] for a in acts], np.int64), ctlD=ctlD)
    c["act_field"] = [a[4] for a in acts]
    M.ctl = c
    return M


def _field_fit(doc, overrides, fields, inits, maxes, i, M):
    """How the step a driven field needs follows the signal: dt = C |x|^-p, one power law for values above the first
    one and one below, each fitted from the model compiled at two values. Returns [C_up, p_up, C_down, p_down]."""
    from .control import with_values
    x0 = inits[i]; sg = -1.0 if x0 < 0 else 1.0; b0 = abs(x0) if x0 else 1.0

    def dt_at(x):
        v = list(inits); v[i] = sg * x
        try:
            Mp = _compile_model(with_values(doc, fields, v), overrides)
        except SpecError:
            return None
        return Mp.dt_auto

    def fit(x1, x2, up):
        d1, d2 = dt_at(x1), dt_at(x2)
        if not d1 or not d2 or x1 <= 0 or x2 <= x1:
            return 0.0, 0.0
        p = math.log(d1 / d2) / math.log(x2 / x1)
        if (up and p < 0.05) or (not up and p > -0.05):
            return 0.0, 0.0                          # this way the field does not set the step
        p = max(-2.0, min(2.0, p))
        return d2 * x2 ** p, p
    mx = abs(maxes[i]) if maxes[i] is not None else None
    if mx is not None:
        cu, pu = fit(mx / 4.0, mx, True) if mx > b0 else (0.0, 0.0)
    else:
        cu, pu = fit(16.0 * b0, 256.0 * b0, True)
    cd, pd = fit(b0 / 16.0, b0, False) if x0 else (0.0, 0.0)
    return [cu, pu, cd, pd]


def _control_notes(M, R, SGT):
    """After a run: driven fields that went past their max, or that move the stability limit and have no max."""
    c = getattr(M, "ctl", None)
    if not c or not getattr(M, "ctl_fields", None) or getattr(M, "adaptive", False):
        return                                       # (an adaptive step follows the field: nothing to flag)
    for a, fi in enumerate(c["act_field"]):
        fd = M.ctl_fields[fi]; j = int(c["asig"][a])
        hi = float(SGT[j, 1]) if SGT.shape[0] > j and SGT[j, 1] > -1e299 else fd["init_v"]
        key = fd["where"].replace(".", "__")
        if fd["max_v"] is not None and hi > fd["max_v"] * (1 + 1e-9) + 1e-300:
            R[f"_ctl_over_max__{key}"] = hi
        elif fd["max_v"] is None and fd["sensitive"] and hi > fd["init_v"] * (1 + 1e-6):
            R[f"_ctl_no_max__{key}"] = hi


def control_messages(R):
    """Plain-language notes for the flags _control_notes leaves in a run's outputs."""
    out = []
    for k, v in R.items():
        if k.startswith("_ctl_over_max__"):
            out.append(f"{k[15:].replace('__', '.')} was driven to {v:.4g}, past its max: the time step was sized for "
                       "max, so raise max (and leave the time step on auto) or limit the signal")
        elif k.startswith("_ctl_no_max__"):
            out.append(f"{k[13:].replace('__', '.')} was driven up to {v:.4g} but has no max: the time step was sized "
                       f"for its first-step value. Add max: {v:.4g} (or more) to the field so the auto time step stays "
                       "stable at it")
    return out


def sensor_slot(M, name, qty, where):
    """(source, index) the engine reads a body or element quantity from: the same slots outputs use."""
    dim = getattr(M, "dim", "1d")
    if name in M.body_names:
        if dim == "3d":
            from .model3d import body_slot3
            return body_slot3(M, name, qty, where)
        if dim == "2d":
            from .model2d import body_slot
            return body_slot(M, name, qty, where)
        s = {"position": 0, "velocity": 1}.get(qty)
        if s is None:
            raise SpecError(f"a 1d body gives position or velocity, not '{qty}'", where)
        return s, M.body_names.index(name)
    known = set(M.el_names) | set((getattr(M, "groups", None) or {}))
    if name not in known:
        raise SpecError(f"'{name}' is not a body or an element", where)
    qn = QALIAS.get(qty, qty)
    i = el_index(M, name, qn)
    if qn == "force":
        return 2, i
    if qn == "measure":
        return 3, i
    if qn in AUX:
        return 4 + AUX[qn], i
    raise SpecError(f"'{qty}' cannot be read while the model runs (use force, measure or a state such as "
                    "altitude, current, throttle)", where)


def _compile_model(doc, overrides=None):
    f = doc.get("__file__", "")
    overrides = dict(overrides or {})
    try:
        ctx = build_context(doc, overrides)
    except SpecError as exc:
        if not str(exc).startswith(f) or not f:
            raise SpecError(str(exc), "", None, f) if f and f not in str(exc) else exc
        raise
    from .expand import expand
    from .circuit import compile_circuit
    user_doc = doc
    circ, circ_els = compile_circuit(doc, ctx)
    if circ_els:                              # circuit motors join the elements (hidden, one per motor)
        import copy as _copy
        d2 = _copy.copy(doc); d2.lines = getattr(doc, "lines", {})
        d2["elements"] = {**(doc.get("elements") or {}), **circ_els}
        doc = d2
    doc, M_groups = expand(doc, ctx)
    M = Compiled(); M.doc = user_doc; M.cdoc = doc; M.groups = M_groups; M.ctx = ctx; M.dim = doc.get("dim", "1d")
    M.circ = circ
    if M.dim == "2d":
        from .model2d import compile2d
        compile2d(doc, ctx, M, overrides)
        return _run_settings(M, doc, ctx)
    if M.dim == "3d":
        from .model3d import compile3d
        compile3d(doc, ctx, M, overrides)
        return _run_settings(M, doc, ctx)
    bodies = doc["bodies"]
    names = [b for b, d in bodies.items() if not d.get("ground")]
    M.body_names = names
    ground = [b for b, d in bodies.items() if d.get("ground")]
    nb = len(names)
    m = np.zeros(nb); q0 = np.zeros(nb); v0 = np.zeros(nb); dof = []
    for i, b in enumerate(names):
        d = bodies[b]; where = f"bodies.{b}"
        dof.append(d["dof"])
        m[i] = num(resolve(d["mass"] if d["dof"] == "translation" else d["inertia"], ctx, where), where)
        q0[i] = num(resolve(d.get("x0", 0.0), ctx, where + ".x0"), where + ".x0")
        v0[i] = num(resolve(d.get("v0", 0.0), ctx, where + ".v0"), where + ".v0")
        if m[i] <= 0:
            raise SpecError("mass/inertia must be positive", where, line_of(bodies, b), f)
    M.dof = dof
    els = doc["elements"]; enames = list(els)
    ne = len(enames)
    typ = np.zeros(ne, np.int64); one = np.zeros(ne, np.int64); act = np.ones(ne, np.int64)
    P = np.zeros((ne, E.NP)); C = np.zeros((ne, nb)); s0 = np.zeros(ne)
    tabs_x, tabs_F, tab_k = [], [], np.zeros(ne)
    M.el_info = {}
    for e, en in enumerate(enames):
        d = els[en]; where = f"elements.{en}"; t = d["type"]
        typ[e] = TYPE_CODE[t]; one[e] = 1 if d.get("one_sided") else 0
        try:
            coef, const = eval_measure(d["measure"], names + ground, numeric_namespace(ctx))
        except ExprError as exc:
            raise SpecError(f"measure: {exc}", where, line_of(d, "measure"), f)
        for b, c in coef.items():
            if b in names:
                C[e, names.index(b)] = c
        s0[e] = const
        g = lambda key, default=None: (num(resolve(d[key], ctx, f"{where}.{key}"), f"{where}.{key}")
                                       if key in d else default)
        tx = np.zeros(0); tF = np.zeros(0)
        if t == "spring":
            P[e, 0] = g("k")
        elif t == "dashpot":
            P[e, 0] = g("c")
        elif t == "kelvin_voigt":
            P[e, 0] = g("k"); P[e, 1] = g("c")
        elif t == "elastoplastic":
            tb = resolve(d["table"], ctx, f"{where}.table")
            if not isinstance(tb, dict) or not {"k", "acc_tab", "F_tab"} <= set(tb):
                raise SpecError("table must give k, acc_tab and F_tab (e.g. a beam_plastic_spring or "
                                "plastic_indentation_table block, or a curve_to_spring_table result)",
                                where, line_of(d, "table"), f)
            tab_k[e] = float(tb["k"]); P[e, 0] = tab_k[e]
            tx = np.ascontiguousarray(tb["acc_tab"], float); tF = np.ascontiguousarray(tb["F_tab"], float)
            if overrides.get(ELASTIC + en) or overrides.get(ELASTIC + en.split("__")[0]):
                tF = tF * 0 + 1e30
            P[e, 1] = g("break_at", math.inf)
        elif t == "hertz":
            P[e, 0] = g("K"); P[e, 1] = g("n", 1.5); P[e, 2] = g("alpha", 0.0)
        elif t == "friction":
            P[e, 0] = g("F_c"); P[e, 1] = g("v_eps"); P[e, 2] = g("F_s", 0.0); P[e, 3] = g("v_s", 0.0)
        elif t == "drag":
            P[e, 0] = g("K")
        elif t == "impact":                      # rigid impact: restitution e, Coulomb mu, rest speed v_rest
            P[e, 0] = g("e"); P[e, 1] = g("mu", 0.0); P[e, 2] = g("v_rest", 0.0)
            if not 0.0 <= P[e, 0] <= 1.0 or P[e, 1] < 0.0 or P[e, 2] < 0.0:
                raise SpecError("impact needs 0 <= e <= 1, mu >= 0 and v_rest >= 0", where, line_of(d, "e"), f)
        elif t == "force":
            if ("P" in d) + ("csv" in d) + ("pulse" in d) != 1:
                raise SpecError("a force needs one of P (a constant), csv (a table against time or the measure) or pulse "
                                "(a standard shape: half_sine, triangle, sine...)", where, line_of(d, "type"), f)
            if "pulse" in d:
                from .loadtable import compile_pulse
                typ[e], tx, tF = compile_pulse(e, d, ctx, where, doc, P, tab_k)
            elif "csv" in d:
                from .loadtable import compile_load
                typ[e], tx, tF = compile_load(e, d, ctx, where, doc, P, tab_k)
                from .loadtable import check_load_kind
                nzb = [i for i in range(nb) if C[e, i] != 0.0]
                check_load_kind(d, bool(nzb) and all(dof[i] == "rotation" for i in nzb), where, doc)
            else:
                P[e, 0] = g("P")
        elif t == "emotor":                       # a circuit motor: torque from the circuit, speed back to it
            G = g("gear", 1.0)
            P[e, 0] = G; P[e, 1] = g("__idx__"); P[e, 2] = g("__kt__"); P[e, 3] = g("__R__")
            nz = [i for i in range(nb) if C[e, i] != 0.0]
            if len(nz) != 1 or dof[nz[0]] != "rotation":
                raise SpecError("a circuit motor's measure must be one rotation body (the shaft it turns)", where,
                                line_of(d, "measure"), f)
            m[nz[0]] += g("J_rotor", 0.0) * G * G
        elif t == "drive":
            kv = g("kv"); G = g("gear", 1.0)
            P[e, :10] = [1.0 / kv, g("R"), g("I_max"), G, g("cells"), g("R0"), g("R1", 0.0), g("C1", 0.0),
                       g("capacity"), g("throttle", 1.0)]
            if not 0.0 <= P[e, 9] <= 1.0:
                raise SpecError("throttle is a fraction of full duty, 0 to 1", where, line_of(d, "throttle"), f)
            Jr = g("J_rotor", 0.0)
            nz = [i for i in range(nb) if C[e, i] != 0.0]
            if len(nz) != 1 or dof[nz[0]] != "rotation":
                raise SpecError("a drive's measure must be one rotation body (the shaft it turns)", where,
                                line_of(d, "measure"), f)
            m[nz[0]] += Jr * G * G
        if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
            act[e] = 0
        tabs_x.append(tx); tabs_F.append(tF)
        me = g("mass", 0.0)
        if me < 0:
            raise SpecError("an element's mass must be >= 0", where, line_of(d, "mass"), f)
        if me > 0:
            # the element's own mass, lumped onto the bodies it joins: half to each moving end (a segment of a chain,
            # or a free-free spring: momentum kept); a third when the other end is ground (Rayleigh: the spring's
            # mass seen at its moving end). A lever factor c puts c^2 m on a rotation (inertia at that radius).
            nzb = [i for i in range(nb) if C[e, i] != 0.0]
            frac = 0.5 if (d.get("__lump__") in ("half", "ends") or len(nzb) == 2) else (1.0 / 3.0 if len(nzb) == 1 else 1.0 / len(nzb))
            for i in nzb:
                if d.get("__lump__") == "ends" and "__" in names[i]:
                    continue                      # chain nodes already carry their share
                m[i] += frac * me * C[e, i] ** 2
        M.el_info[en] = dict(type=t, measure=str(d["measure"]), params=P[e].copy(), mass=me, k=tab_k[e] if t == "elastoplastic"
                             else (P[e, 0] if t in ("spring", "kelvin_voigt") else None))
    off = np.zeros(ne, np.int64); ln = np.zeros(ne, np.int64); pos = 0
    for e in range(ne):
        off[e] = pos; ln[e] = len(tabs_x[e]); pos += ln[e]
    M.arrays = dict(q0=q0, v0=v0, m=m, typ=typ, one_sided=one, active=act, P=P, C=C, s0=s0, tab_off=off,
                    tab_len=ln, tab_k=tab_k,
                    tab_x=np.concatenate(tabs_x) if pos else np.zeros(1),
                    tab_F=np.concatenate(tabs_F) if pos else np.zeros(1))
    M.el_names = enames
    return _run_settings(M, doc, ctx)


def _run_settings(M, doc, ctx):
    run = doc["run"]
    M.t_end = num(resolve(run["t_end"], ctx, "run.t_end"), "run.t_end")
    M.dt_crit = critical_dt(M)
    if getattr(M, "circ", None):                # each circuit motor's element, by motor number
        M.circ["mot_el"] = np.array([M.el_names.index(f"__circ_{pn}") for pn in M.circ["motors"]] or [0], np.int64)
    dt = run.get("dt", "auto")
    adaptive = isinstance(dt, str) and dt.strip().lower() == "adaptive"
    if adaptive:                                 # the free-motion limit: every element but the contacts
        keep = (getattr(M, "dt_crit_beam", None), getattr(M, "dt_crit_other", None))
        M.contact_el = contact_mask(M)
        crit_free = critical_dt(M, M.contact_el) if M.contact_el.any() else M.dt_crit
        free_bo = (getattr(M, "dt_crit_beam", math.inf), getattr(M, "dt_crit_other", None))
        M.dt_crit_beam, M.dt_crit_other = keep
    base = _step_share(M, M.dt_crit, getattr(M, "dt_crit_beam", math.inf), getattr(M, "dt_crit_other", None))
    if adaptive:
        # free flight under steady forces is first order in the step (semi-implicit Euler): a finer share of t_end
        # than the fixed automatic step uses keeps positions after a long flight within about 1 %
        M.dt_auto_free = _step_caps(M, _step_share(M, crit_free, *free_bo))
        if M.contact_el.any():
            M.dt_auto_free = min(M.dt_auto_free, M.t_end / 20000.0)
    M.dt_auto = _step_caps(M, base)
    if adaptive:
        M.dt_auto_free = max(M.dt_auto_free, M.dt_auto)
    M.timeout = parse_timeout(run.get("timeout", "auto"), ctx)
    M.adaptive = adaptive
    lo = num(resolve(run["dt_min"], ctx, "run.dt_min"), "run.dt_min") if run.get("dt_min") not in (None, "") else None
    hi = num(resolve(run["dt_max"], ctx, "run.dt_max"), "run.dt_max") if run.get("dt_max") not in (None, "") else None
    if adaptive:
        # steps between dt_contact (contacts touching or about to) and dt_free (free motion), inside the run bounds
        def clamp(x):
            x = min(x, hi) if hi is not None else x
            return max(x, lo) if lo is not None else x
        M.dt_contact = clamp(M.dt_auto); M.dt_free = max(clamp(M.dt_auto_free), M.dt_contact)
        M.dt_floor = lo if lo is not None else M.dt_contact * 1e-4
        M.dt = M.dt_contact
    elif dt == "auto":
        # automatic step, optionally held inside user bounds (run.dt_min / run.dt_max)
        M.dt = M.dt_auto
        if hi is not None:
            M.dt = min(M.dt, hi)
        if lo is not None:
            M.dt = max(M.dt, lo)
    else:
        M.dt = num(resolve(dt, ctx, "run.dt"), "run.dt")
    return M


def _step_share(M, crit, cb, other):
    """explicit-step share: 2% of the stability limit for contacts and springs (impact accuracy); beams are smooth
    structural elements and run at 20% of theirs (their high modes need stability, not resolution)"""
    cb = math.inf if cb is None else cb
    return min(0.02 * crit if not math.isfinite(cb) or crit < cb else
               min(0.02 * (other if other is not None else math.inf), 0.2 * cb), M.t_end / 2000.0)


MOTOR_SHARE = 1.0           # a step per drive time constant: its own response is exact at any step, and one tau
                            # keeps what it couples to (loads, friction) within about 0.2 %


def _step_caps(M, dt_auto):
    """The automatic step's other limits: load features, free rotation, vehicles, circuit transients."""
    M.dt_auto = dt_auto
    a_ = getattr(M, "arrays", None)       # loads against time (CSV tables, standard shapes) must be resolved too
    if a_ is not None and "typ" in a_:
        feats = [a_["P"][e, 3] for e in range(len(a_["typ"])) if a_["typ"][e] == 13 and a_["P"][e, 1] == 0.0 and a_["P"][e, 3] > 0]
        if feats:
            M.dt_auto = min(M.dt_auto, min(feats))
    if getattr(M, "dim", "1d") == "3d":
        from .model3d import rotation_dt3d
        M.dt_auto = min(M.dt_auto, rotation_dt3d(M))
    if math.isfinite(getattr(M, "dt_motor", math.inf)):    # a drive's time constant: resolved, not just stable
        M.dt_auto = min(M.dt_auto, MOTOR_SHARE * M.dt_motor)
    a2 = getattr(M, "arrays", None)
    if a2 is not None and "typ" in a2 and np.any((a2["typ"] >= 18) & (a2["typ"] <= 22)):
        # a vehicle in flight: resolve the engines' rise and tail-off, and never more than 50 ms a step
        ts = [a2["P"][e, k] for e in range(len(a2["typ"])) if a2["typ"][e] == 18 for k in (5, 6) if a2["P"][e, k] > 0]
        M.dt_auto = min(M.dt_auto, 0.05, *(x / 10.0 for x in ts))
    if getattr(M, "circ", None):                # the circuit's fastest transient, five steps a time constant
        from .circuit import circuit_tau
        M.dt_circ = circuit_tau(M.circ)
        if math.isfinite(M.dt_circ):
            M.dt_auto = min(M.dt_auto, 0.2 * M.dt_circ)
    return M.dt_auto


def contact_mask(M):
    """1 for each contact element: one-sided (touches only when pressed), hertz, or a geometric contact measure.
    These are the elements the adaptive step leaves out in free motion."""
    a = M.arrays; ne = len(M.el_names)
    out = np.zeros(ne, np.int64)
    dim = getattr(M, "dim", "1d")
    geo = ()
    if dim == "2d":
        from . import engine2d as _E
        geo = (_E.MK_EDGE, _E.MK_CIRC)
    elif dim == "3d":
        from . import engine3d as _E
        geo = (_E.MK_PLANE, _E.MK_SPH)
    for e in range(ne):
        t = int(a["typ"][e])
        if t in (10, 14, 15):                      # friction, rigid impacts, breakaways: no step of their own
            continue
        if a["one_sided"][e] == 1 or t == 4 or ("mk" in a and int(a["mk"][e]) in geo):
            out[e] = 1
    return out


def critical_dt(M, skip=None):
    """Stability limit of the explicit step. skip: elements to leave out (1 = skip), for the adaptive free step."""
    if getattr(M, "dim", "1d") == "2d":
        from .model2d import critical_dt2d
        return critical_dt2d(M, skip)
    if getattr(M, "dim", "1d") == "3d":
        from .model3d import critical_dt3d
        return critical_dt3d(M, skip)
    a = M.arrays; best = math.inf; M.dt_motor = math.inf
    for e in range(len(M.el_names)):
        if skip is not None and skip[e]:
            continue
        t = a["typ"][e]
        k = a["tab_k"][e] if t in (3, 13) else (a["P"][e, 0] if t in (0, 2) else 0.0)
        c = a["C"][e]
        inv = sum(c[i] ** 2 / a["m"][i] for i in range(len(c)) if c[i] != 0.0)
        if t == 8 or t == 17:            # motor: mechanical time constant J R / (G Kt)^2, stable below 2 tau
            Kt, R, G = ((a["P"][e, 0], a["P"][e, 1], a["P"][e, 3]) if t == 8 else
                        (a["P"][e, 2], a["P"][e, 3], a["P"][e, 0]))
            if inv > 0:
                lim = 2.0 * R / (inv * (G * Kt) ** 2)
                # a drive steps itself exactly (engine1d _force) and a circuit motor is solved with its shaft
                # (circuit_step): neither is a stability limit. Its time constant only sets the accuracy cap, and the
                # engine needs its inertia (P[e, 10] = 1 / inertia along the measure)
                a["P"][e, 10] = inv
                M.dt_motor = min(M.dt_motor, 0.5 * lim)
            continue
        if k > 0 and inv > 0:
            best = min(best, 2.0 / math.sqrt(k * inv))
    return best if math.isfinite(best) else M.t_end / 100.0


# ------------------------------------------------------------------------------------------- run
CIRC_COL = {"current": 0, "voltage": 1, "power": 2, "energy": 3, "soc": 4, "state": 4, "torque": 5,
            "temperature": 6, "heat": 7}


def circuit_args(M):
    """The circuit arrays the engines take (empty when the model has no circuit)."""
    c = getattr(M, "circ", None)
    if not c:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 12)),
                np.zeros((0, 6)), np.zeros((0, 8)), 0, np.zeros(1, np.int64))
    a = c["arrays"]
    return (a["ptype"], a["na"], a["nb"], a["prm"], a["pst0"], a["th"], int(a["nn"]), c["mot_el"])


def _ctl_history(M, H, LS):
    c = getattr(M, "ctl", None)
    if c:
        for j, nm in enumerate(c["names"]):
            H[f"control.{nm}"] = LS[:, j] if LS.shape[1] > j else np.zeros(LS.shape[0])


def _circ_history(M, H, LC):
    c = getattr(M, "circ", None)
    if not c or LC.shape[1] < 8:
        return
    for k, pn in enumerate(c["names"]):
        for q, col in CIRC_COL.items():
            H[f"{pn}.{q}"] = LC[:, k * 8 + col]


def _group_history(M, H):
    for name, g in (getattr(M, "groups", None) or {}).items():
        for key in [k for k in list(H) if k.startswith(g.segs[0] + ".")]:
            H[name + key[len(g.segs[0]):]] = H[key]
        if g.probe and f"{g.probe}.measure" in H:
            H[f"{name}.measure"] = H[f"{g.probe}.measure"]


def el_index(M, name, qn="force"):
    """Engine row of an element. An expanded element (segments) answers measure from its probe on the original
    measure and everything else from the segment at its positive end."""
    g = (getattr(M, "groups", None) or {}).get(name)
    if g is None:
        return M.el_names.index(name)
    if qn == "measure" and g.probe:
        return M.el_names.index(g.probe)
    return M.el_names.index(g.segs[0])


def _crossings(M):
    doc = M.doc; outs = doc["outputs"]
    src, idx, thr, dr, keys = [], [], [], [], []
    for name, o in outs.items():
        red = o.get("reduce", "final")
        if not red.startswith("first_time"):
            continue
        where = f"outputs.{name}"
        val = num(resolve(o["value"], M.ctx, where + ".value"), where + ".value")
        if "control" in o:
            s, i = 50, list(M.ctl["names"]).index(o["control"])
        elif "body" in o and getattr(M, "dim", "1d") == "3d":
            from .model3d import body_slot3
            if o["quantity"] == "kinetic_energy":
                raise SpecError("first_time_* does not work on kinetic_energy", where)
            s, i = body_slot3(M, o["body"], o["quantity"], where)
        elif "body" in o and getattr(M, "dim", "1d") == "2d":
            from .model2d import body_slot
            if o["quantity"] == "kinetic_energy":
                raise SpecError("first_time_* does not work on kinetic_energy", where)
            s, i = body_slot(M, o["body"], o["quantity"], where)
        elif "body" in o:
            i = M.body_names.index(o["body"])
            s = {"position": 0, "velocity": 1}.get(o["quantity"])
            if s is None:
                raise SpecError("first_time_* works on position or velocity of a body", where)
        else:
            qn = QALIAS.get(o["quantity"], o["quantity"]); i = el_index(M, o["element"], qn)
            if qn == "force":
                s = 2
            elif qn == "measure":
                s = 3
            elif qn in AUX:
                s = 4 + AUX[qn]
            else:
                raise SpecError(f"first_time_* cannot use {qn}", where)
        src.append(s); idx.append(i); thr.append(val); dr.append(1 if red.endswith("above") else -1); keys.append(name)
    return (np.array(src, np.int64), np.array(idx, np.int64), np.array(thr, float), np.array(dr, np.int64), keys)


# run.timeout: auto -> AUTO_FACTOR x the estimated run time (never under AUTO_MIN_S). The estimate is deliberately
# rudimentary: steps x elements / RATE, where RATE (element-steps per second) is learned from finished runs.
RATE = [5.0e6]
AUTO_FACTOR = 5.0
AUTO_MIN_S = 10.0


def estimate_run_s(M, dt=None):
    """Rough wall-clock time of one run on this computer."""
    n = M.t_end / (dt or M.dt)
    return n * max(1, len(M.el_names)) / RATE[0]


def auto_timeout(M, dt=None):
    return max(AUTO_MIN_S, AUTO_FACTOR * estimate_run_s(M, dt))


def parse_timeout(v, ctx=None):
    """run.timeout: 'auto' (5x the estimated run time, at least 10 s), 'none' / 'off' (never), or a time
    ('90 s', '5 min', 120). Returns 'auto', seconds, or None."""
    if v is None or (isinstance(v, str) and v.strip().lower() == "auto"):
        return "auto"
    if isinstance(v, str) and v.strip().lower() in ("none", "off", "never"):
        return None
    t = num(resolve(v, ctx or {}, "run.timeout"), "run.timeout")
    if t <= 0:
        raise SpecError("run.timeout must be a positive time, auto, or none", "run.timeout")
    return float(t)


def adaptive_args(M, dt=None):
    """The engines' adaptive-step arrays (see adaptive.py). dt: a step asked for instead of the model's own (the
    dt-convergence check halves it); every adaptive step is scaled by dt / M.dt."""
    from .adaptive import fixed_adaptive, NADP
    na = len(M.ctl["ae"]) if getattr(M, "ctl", None) else 0
    ne = len(M.el_names)
    if not getattr(M, "adaptive", False):
        return fixed_adaptive(ne, na)
    sc = dt / M.dt if dt else 1.0
    adp = np.zeros(NADP)
    adp[0] = 1.0; adp[1] = M.dt_free * sc; adp[2] = M.dt_contact * sc
    adp[3] = 4.0 * adp[1]                         # look-ahead: a contact closing within four free steps
    adp[4] = 1.25; adp[5] = M.t_end; adp[6] = min(M.dt_floor * sc, adp[2]); adp[7] = M.t_end / 800.0
    adp[11] = 1e300
    adf = np.zeros((na, 4))
    fits = getattr(M, "dt_fits", None)
    if fits is not None and len(fits) == na:
        adf[:] = fits
        adf[:, 0] *= sc; adf[:, 2] *= sc
    return adp, adf, np.asarray(M.contact_el, np.int64)


def run_model(M, dt=None, history=False, n_log=800, timeout="model", _n=None):
    """timeout: 'model' uses run.timeout; None runs to the end; a number is seconds of wall-clock time."""
    import threading
    a = M.arrays
    adp, adf, adc = adaptive_args(M, dt)
    dt = dt or M.dt
    n = int(round(M.t_end / dt)) if _n is None else _n
    if adp[0] > 0 and _n is None:
        n = int(M.t_end / adp[6]) + 10           # the most steps it could take (every step at the floor)
    cs, ci, ct, cd, ckeys = _crossings(M)
    wants_at = any(o.get("reduce") in ("at", "time_of_max", "time_of_min") for o in (M.doc.get("outputs") or {}).values())
    if wants_at:
        n_log = max(n_log, 4000)
    every = max(n // n_log, 1) if (history or wants_at) else 0
    if adp[0] > 0:
        every = 1 if (history or wants_at) else 0
        adp[7] = M.t_end / n_log                 # log by time: the steps are not all the same length
    limit = getattr(M, "timeout", "auto") if timeout == "model" else timeout
    if limit == "auto":
        limit = auto_timeout(M, dt)
    stop = np.zeros(3, dtype=np.int64)            # [stop request, steps run, 1 = ended itself (touchdown)]
    dim2 = getattr(M, "dim", "1d") == "2d"
    dim3 = getattr(M, "dim", "1d") == "3d"
    if limit:
        # the first run after an install compiles the engine (tens of seconds); do that before the timeout clock
        # starts, with a zero-step run, so the compile is never mistaken for a slow simulation
        from . import engine2d as _E2, engine3d as _E3
        eng = _E3.run_engine3d if dim3 else (_E2.run_engine2d if dim2 else E.run_engine)
        if not eng.signatures:
            run_model(M, dt=dt, history=False, n_log=n_log, timeout=None, _n=0)
    timer = threading.Timer(limit, stop.__setitem__, (0, 1)) if limit else None
    if timer is not None:
        timer.daemon = True; timer.start()
    import time as _time
    t_run = _time.perf_counter()
    Wg = 0.0; SP = None
    cargs = circuit_args(M)
    from .control import control_args
    kargs = control_args(M)
    try:
        if dim3:
            from . import engine3d as E3
            (FB, quat, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, tcr, LT, LB, LE, Fe, se, Wg, Lw, KEpk, CP, CTR, LC, SGF, SGT, LS) = E3.run_engine3d(
                n, dt, a["X0"], a["V0"], a["quat0"], a["Lw0"], a["wsp0"], a["th0"], a["minv"], a["rmode"], a["Iinv"],
                a["axis"], a["Iax"], a["mass"], a["com"], a["grav"], a["typ"], a["one_sided"], a["active"], a["mk"],
                a["ba"], a["bb"], a["ref"], a["P"], a["C"], a["s0"], a["GA"], a["GB"], a["GX"],
                a["tab_off"], a["tab_len"], a["tab_k"], a["tab_x"], a["tab_F"], every, cs, ci, ct, cd, *cargs, *kargs,
                adp, adf, adc, stop)
        elif dim2:
            from . import engine2d as E2
            (q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, SP, aux, tcr, LT, LB, LE, Fe, se, Wg, KEpk, CP, CTR, LC, SGF, SGT, LS) = E2.run_engine2d(
                n, dt, a["q0"], a["v0"], a["minv"], a["mass"], a["com"], a["grav"], a["typ"], a["one_sided"],
                a["active"], a["mk"], a["ba"], a["bb"], a["ref"], a["P"], a["C"], a["s0"], a["GA"], a["GB"], a["GX"],
                a["tab_off"], a["tab_len"], a["tab_k"], a["tab_x"], a["tab_F"], every, cs, ci, ct, cd, *cargs, *kargs,
                a["planet"], adp, adf, adc, stop)
        else:
            (q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, tcr, LT, LB, LE, Fe, se, KEpk, CP, CTR, LC, SGF, SGT, LS) = E.run_engine(
                n, dt, a["q0"], a["v0"], a["m"], a["typ"], a["one_sided"], a["active"], a["P"], a["C"], a["s0"],
                a["tab_off"], a["tab_len"], a["tab_k"], a["tab_x"], a["tab_F"], every, cs, ci, ct, cd, *cargs, *kargs,
                adp, adf, adc, stop)
    finally:
        if timer is not None:
            timer.cancel()
    timed_out = int(stop[1]) < n and int(stop[2]) == 0
    secs = _time.perf_counter() - t_run
    if secs > 0.05 and int(stop[1]) > 0:          # learn this computer's speed for the next estimate
        RATE[0] = 0.5 * RATE[0] + 0.5 * int(stop[1]) * max(1, len(M.el_names)) / secs
    # energy ledger
    glob = (KE1 - KE0) + A.sum() - Wg
    internal = 0.0; inputs = max(Wg, 0.0)
    for e in range(len(A)):
        if IN[e] < 0:
            inputs += -A[e]
            continue
        inputs += IN[e]
        if D[e] >= 0:
            internal += abs(A[e] - ((S1[e] - S0[e]) + D[e] - IN[e]))
    imp = [e for e in range(len(A)) if a["typ"][e] == 14]
    hits = float(sum(aux[e, 3] for e in imp)) if imp else 0.0      # energy carried into rigid impacts
    # the yardstick is the largest energy seen in the run: peak kinetic energy (a pendulum ending at the top of a
    # swing has none left), stored energy, work in and out, and a floor of every body dropping 1 mm under gravity
    # (a rigidly held body never moves, so its tiny absolute error must not read as 100 %)
    gfloor = 0.0
    if dim2 or dim3:
        g = float(np.linalg.norm(a["grav"])); gfloor = float(np.sum(a["mass"])) * g * 1e-3
    scale = max(KE0, KE1, KEpk, inputs, abs(Wg), np.abs(A).sum(), float(np.abs(S1).sum()), hits, gfloor, 1e-30)
    led = (abs(glob) + internal) / scale
    R = dict(_ledger_rel=float(led))
    if adp[0] > 0:                                # what the adaptive step did
        R.update(_steps=float(adp[10]), _dt_smallest=float(adp[11]) if adp[11] < 1e299 else float(adp[2]),
                 _dt_largest=float(adp[12]), _contact_steps=float(adp[13]))
    if getattr(M, "circ", None) and CP.shape[0]:        # the circuit's own books: part energies sum to zero
        tot = float(np.abs(CP[:, 3]).sum())
        R["_circuit_ledger_rel"] = float(abs(CP[:, 3].sum()) / tot) if tot > 0 else 0.0
    if dim2 or dim3:
        R["_gravity_work"] = float(Wg)
    AT = _at_values(M, LT if every else LT[:0], LB, LE, dim2, LS) if any(o.get("reduce") in ("at", "time_of_max", "time_of_min") for o in M.doc["outputs"].values()) else {}
    if timed_out:
        R["_timed_out"] = 1.0
        R["_t_stopped"] = float(adp[14])
        R["_timeout_s"] = float(limit)
    outs = M.doc["outputs"]
    tc = dict(zip(ckeys, tcr))
    RSI = {}
    for name, o in outs.items():
        if "expr" in o:
            continue
        red = o.get("reduce", "final")
        pk = None
        if "control" in o:
            jn = list(M.ctl["names"]).index(o["control"])
            if red in ("at", "time_of_max", "time_of_min"):
                val = AT[name]
            elif red.startswith("first_time"):
                val = tc[name]
            else:
                lo, hi, fin = SGT[jn, 0], SGT[jn, 1], SGF[jn]
                if lo > 1e299:
                    lo = hi = fin
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]
                pk = max(abs(lo), abs(hi))
        elif "part" in o:
            k_ = M.circ["names"].index(o["part"]); col = CIRC_COL[o["quantity"]]
            lo, hi, fin = CTR[k_, 2 * col], CTR[k_, 2 * col + 1], CP[k_, col]
            if lo > 1e299:
                lo = hi = fin
            val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]; pk = max(abs(lo), abs(hi))
        elif "body" in o and dim3:
            qn = o["quantity"]; i = M.body_names.index(o["body"])
            if qn == "kinetic_energy":
                if red != "final":
                    raise SpecError("kinetic_energy supports reduce: final only", f"outputs.{name}")
                from .engine3d import NBQ
                mi = a["minv"][i]; vb = FB[i * NBQ + 3:i * NBQ + 6]
                val = sum(0.5 * vb[k] ** 2 / mi[k] for k in range(3) if mi[k] > 0)
                if a["rmode"][i] == 2:
                    val += 0.5 * float(np.dot(FB[i * NBQ + 9:i * NBQ + 12], Lw[i]))
                elif a["rmode"][i] == 1 and a["Iax"][i] > 0:
                    val += 0.5 * FB[i * NBQ + 8] ** 2 / a["Iax"][i]
            elif red.startswith("first_time"):
                val = tc[name]
            else:
                from .model3d import body_slot3
                _, k = body_slot3(M, o["body"], qn, f"outputs.{name}")
                lo, hi, fin = BT[k, 0], BT[k, 1], FB[k]
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]; pk = max(abs(lo), abs(hi))
        elif "body" in o and dim2:
            qn = o["quantity"]
            if qn == "kinetic_energy":
                if red != "final":
                    raise SpecError("kinetic_energy supports reduce: final only", f"outputs.{name}")
                i = M.body_names.index(o["body"]); mi = a["minv"][3 * i:3 * i + 3]
                val = sum(0.5 * v[3 * i + k] ** 2 / mi[k] for k in range(3) if mi[k] > 0)
            elif red in ("at", "time_of_max", "time_of_min"):
                val = AT[name]; pk = AT.get("__range__" + name)
            elif red.startswith("first_time"):
                val = tc[name]
            else:
                from .model2d import body_slot
                src, k = body_slot(M, o["body"], qn, f"outputs.{name}")
                if src == 8:
                    lo, hi, fin = SP[k, 0], SP[k, 1], math.hypot(v[3 * k], v[3 * k + 1])
                elif src == 0:
                    lo, hi, fin = BT[k, 0], BT[k, 1], q[k]
                else:
                    lo, hi, fin = BT[k, 2], BT[k, 3], v[k]
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]; pk = max(abs(lo), abs(hi))
        elif "body" in o:
            i = M.body_names.index(o["body"]); qn = o["quantity"]
            if qn == "kinetic_energy":
                if red != "final":
                    raise SpecError("kinetic_energy supports reduce: final only", f"outputs.{name}")
                val = 0.5 * a["m"][i] * v[i] ** 2
            elif red in ("at", "time_of_max", "time_of_min"):
                val = AT[name]; pk = AT.get("__range__" + name)
            elif red.startswith("first_time"):
                val = tc[name]
            else:
                lo, hi, fin = (BT[i, 0], BT[i, 1], q[i]) if qn == "position" else (BT[i, 2], BT[i, 3], v[i])
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]; pk = max(abs(lo), abs(hi))
        else:
            qn = o["quantity"]
            grp = (getattr(M, "groups", None) or {}).get(o["element"])
            rows = [M.el_names.index(x) for x in grp.segs] if grp is not None else [M.el_names.index(o["element"])]
            if grp is not None and qn == "measure" and grp.probe:
                rows = [M.el_names.index(grp.probe)]
            joint = grp is not None and grp.kind in ("weld", "hinge", "slider")
            if joint and qn in ("force", "moment"):
                # a joint is several engine rows: force is the push / pull rows (pin, or the slider's sideways
                # guides, as one magnitude); moment is the turn lock. Never mixed (N and N m)
                tags = ("__rot",) if qn == "moment" else ("__pin", "__g1", "__g2")
                rows = [M.el_names.index(x) for x in grp.segs if x.endswith(tags)]
                if qn == "moment" and not rows:
                    raise SpecError(f"'{o['element']}' has no turn lock, so no moment (give k_rot or rigid)",
                                    f"outputs.{name}")

            def span(e):
                k = {"force": 0, "measure": 1}.get(QALIAS.get(qn, qn))
                if k is not None:
                    lo, hi, fin = TR[e, 2 * k], TR[e, 2 * k + 1], (Fe[e] if k == 0 else se[e])
                else:
                    j = AUX[qn]
                    lo, hi, fin = TR[e, 4 + 2 * j], TR[e, 5 + 2 * j], aux[e, j]
                return (0.0, 0.0, 0.0) if lo > 1e299 else (lo, hi, fin)
            if red in ("at", "time_of_max", "time_of_min"):
                val = AT[name]; pk = AT.get("__range__" + name)
            elif red.startswith("first_time"):
                val = tc[name]
            elif qn == "absorbed_work":
                val = sum(A[e] for e in rows)
            elif qn == "dissipated":
                val = sum((D[e] if D[e] >= 0 else A[e] - (S1[e] - S0[e])) for e in rows)
            else:
                sp = [(lambda e: (0.0, 0.0, 0.0) if TR[e, 0] > 1e299 else (TR[e, 0], TR[e, 1], Fe[e]))(e) for e in rows] \
                    if (joint and qn == "moment") else [span(e) for e in rows]
                if joint and qn in ("force", "moment") and len(sp) > 1:
                    # magnitude of the reaction: final is exact; max is the root sum of each row's peak (an upper
                    # bound when the rows peak at different times)
                    fin = math.sqrt(sum(x[2] ** 2 for x in sp))
                    hi = math.sqrt(sum(max(abs(x[0]), abs(x[1])) ** 2 for x in sp)); lo = 0.0
                    val = {"final": fin, "max": hi, "min": lo, "max_abs": hi}[red]
                elif joint and qn in ("force", "moment") and grp.kind == "slider":
                    x = sp[0]; m_ = max(abs(x[0]), abs(x[1]))
                    val = {"final": abs(x[2]), "max": m_, "min": 0.0, "max_abs": m_}[red]
                else:
                    lo = min(x[0] for x in sp); hi = max(x[1] for x in sp); fin = sp[0][2]
                    val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]; pk = max(abs(lo), abs(hi))
        val = float(val)
        if red.startswith("first_time"):
            reached = val >= 0
            R[f"_{name}_reached"] = 1.0 if reached else 0.0
            if not reached:
                val = M.t_end * 10.0        # sentinel: never crossed within t_end (flagged in reports)
        if o.get("abs"):
            val = abs(val)
        RSI[name] = val                     # expressions read every output in SI, whatever unit it is shown in
        if "unit" in o:
            val = val / unit_factor(o["unit"])
            if pk is not None:
                pk = pk / unit_factor(o["unit"])
        R[name] = val
        if red in ("final", "at") and pk:   # a final or at value is judged against the quantity's own range in Verify
            R[f"_{name}_scale"] = float(pk)
        elif red.startswith("first_time"):  # a crossing time against the run's length (a crossing at t = 0+ is one step)
            R[f"_{name}_scale"] = float(M.t_end) / (unit_factor(o["unit"]) if "unit" in o else 1.0)
    # expression outputs (may use params and other outputs, always in SI)
    for name, o in outs.items():
        if "expr" in o:
            nums = numeric_namespace(M.ctx)
            nums.update({k: x for k, x in R.items() if not k.startswith("_")})
            nums.update(RSI)
            try:
                val = float(eval_numeric(o["expr"], nums))
            except ZeroDivisionError:
                val = math.inf              # e.g. a safety factor when the demand is exactly zero
            except ExprError as exc:
                raise SpecError(str(exc), f"outputs.{name}")
            RSI[name] = val                 # the result is SI; its own unit only changes how it is shown
            R[name] = val / unit_factor(o["unit"]) if "unit" in o else val
    _control_notes(M, R, SGT)
    if history and dim3:
        from .engine3d import NBQ
        from .model3d import BODY_Q3, body_slot3
        H = dict(t=LT)
        for i, b in enumerate(M.body_names):
            for qn, k in BODY_Q3.items():
                H[f"{b}.{qn}"] = LB[:, i * NBQ + k]
            for qn in ("position", "velocity"):
                try:
                    H[f"{b}.{qn}"] = LB[:, body_slot3(M, b, qn, "")[1]]
                except SpecError:
                    pass
        w = 2 + E.NAUX
        for e, en in enumerate(M.el_names):
            H[f"{en}.force"] = LE[:, e * w]; H[f"{en}.measure"] = LE[:, e * w + 1]
            for qn, j in AUX.items():
                H[f"{en}.{qn}"] = LE[:, e * w + 2 + j]
            if M.arrays["typ"][e] == 14:             # rigid impact: running total of the energy it dissipated
                H[f"{en}.dissipated"] = LE[:, e * w + 3]
        _group_history(M, H); _circ_history(M, H, LC); _ctl_history(M, H, LS)
        return R, H
    if history and dim2:
        H = dict(t=LT); nq = 3 * len(M.body_names)
        for i, b in enumerate(M.body_names):
            for k, (qn, vn) in enumerate((("x", "vx"), ("y", "vy"), ("angle", "omega"))):
                H[f"{b}.{qn}"] = LB[:, 3 * i + k]; H[f"{b}.{vn}"] = LB[:, nq + 3 * i + k]
            H[f"{b}.speed"] = np.hypot(H[f"{b}.vx"], H[f"{b}.vy"])
            free = [k for k, fr in enumerate(M.b2[b]["free"]) if fr]
            if len(free) == 1:
                H[f"{b}.position"] = LB[:, 3 * i + free[0]]; H[f"{b}.velocity"] = LB[:, nq + 3 * i + free[0]]
        w = 2 + E.NAUX
        for e, en in enumerate(M.el_names):
            H[f"{en}.force"] = LE[:, e * w]; H[f"{en}.measure"] = LE[:, e * w + 1]
            for qn, j in AUX.items():
                H[f"{en}.{qn}"] = LE[:, e * w + 2 + j]
            for al, base in QALIAS.items():
                H[f"{en}.{al}"] = H[f"{en}.{base}"]
            if M.arrays["typ"][e] == 14:             # rigid impact: running total of the energy it dissipated
                H[f"{en}.dissipated"] = LE[:, e * w + 3]
        _group_history(M, H); _circ_history(M, H, LC); _ctl_history(M, H, LS)
        return R, H
    if history:
        H = dict(t=LT)
        for i, b in enumerate(M.body_names):
            H[f"{b}.position"] = LB[:, 2 * i]; H[f"{b}.velocity"] = LB[:, 2 * i + 1]
        w = 2 + E.NAUX
        for e, en in enumerate(M.el_names):
            H[f"{en}.force"] = LE[:, e * w]; H[f"{en}.measure"] = LE[:, e * w + 1]
            for qn, j in AUX.items():
                H[f"{en}.{qn}"] = LE[:, e * w + 2 + j]
            if M.arrays["typ"][e] == 14:             # rigid impact: running total of the energy it dissipated
                H[f"{en}.dissipated"] = LE[:, e * w + 3]
        _group_history(M, H); _circ_history(M, H, LC); _ctl_history(M, H, LS)
        return R, H
    return R



def _at_values(M, LT, LB, LE, dim2, LS=None):
    """Outputs read from the run's log: reduce: at (the value at a given time, interpolated), time_of_max and
    time_of_min (when the quantity peaks)."""
    out = {}
    if len(LT) == 0:                       # a zero-step run (the engine's first compile): nothing logged
        return {name: 0.0 for name, o in M.doc["outputs"].items() if o.get("reduce") in ("at", "time_of_max", "time_of_min")}
    w = 2 + E.NAUX
    for name, o in M.doc["outputs"].items():
        red = o.get("reduce")
        if red not in ("at", "time_of_max", "time_of_min"):
            continue
        qn = o.get("quantity")
        if "control" in o:
            ser = LS[:, list(M.ctl["names"]).index(o["control"])]
        elif "element" in o:
            e = M.el_names.index(o["element"]); base = QALIAS.get(qn, qn)
            col = {"force": 0, "measure": 1}.get(base)
            ser = LE[:, e * w + col] if col is not None else LE[:, e * w + 2 + AUX[qn]]
        elif dim2:
            i = M.body_names.index(o["body"]); nq = 3 * len(M.body_names)
            if qn == "speed":
                ser = np.hypot(LB[:, nq + 3 * i], LB[:, nq + 3 * i + 1])
            else:
                from .model2d import body_slot
                src, k = body_slot(M, o["body"], qn, f"outputs.{name}")
                ser = LB[:, k] if src == 0 else LB[:, nq + k]
        else:
            i = M.body_names.index(o["body"])
            ser = LB[:, 2 * i] if qn == "position" else LB[:, 2 * i + 1]
        if red == "at":
            t = num(resolve(o["value"], M.ctx, f"outputs.{name}.value"), f"outputs.{name}.value")
            out[name] = float(np.interp(t, LT, ser))
            out["__range__" + name] = float(np.max(np.abs(ser))) if len(ser) else 0.0   # the scale it moved through
        else:
            out[name] = float(LT[int(np.argmax(ser) if red == "time_of_max" else np.argmin(ser))])
    return out

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
             "drag": 6, "force": 7, "drive": 8, "impact": 14}
AUX = {"plastic_set": 0, "broken": 2, "current": 0, "battery_current": 1, "bus_voltage": 2, "soc": 3,
       "moment": 0, "torque": 1, "shear": 2, "hinge_set": 3, "impulse": 0, "friction_impulse": 2, "drift": 3}
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
        elif isinstance(val, str) and _MAT_PROP.match(k):
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
    for k, v in (doc.get("derived", {}) or {}).items():
        try:
            ctx[k] = resolve(v, ctx, f"derived.{k}")
        except SpecError as exc:
            exc.args = (f"{f}:{line_of(doc['derived'], k)} {exc.args[0]}",)
            raise
    return ctx


def compile_model(doc, overrides=None):
    f = doc.get("__file__", "")
    overrides = dict(overrides or {})
    try:
        ctx = build_context(doc, overrides)
    except SpecError as exc:
        if not str(exc).startswith(f) or not f:
            raise SpecError(str(exc), "", None, f) if f and f not in str(exc) else exc
        raise
    from .expand import expand
    user_doc = doc
    doc, M_groups = expand(doc, ctx)
    M = Compiled(); M.doc = user_doc; M.cdoc = doc; M.groups = M_groups; M.ctx = ctx; M.dim = doc.get("dim", "1d")
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
            if ("P" in d) + ("csv" in d) != 1:
                raise SpecError("a force needs P (a constant) or csv (a table against time or the measure), not both",
                                where, line_of(d, "type"), f)
            if "csv" in d:
                from .loadtable import compile_load
                typ[e], tx, tF = compile_load(e, d, ctx, where, doc, P, tab_k)
            else:
                P[e, 0] = g("P")
        elif t == "drive":
            kv = g("kv"); G = g("gear", 1.0)
            P[e, :] = [1.0 / kv, g("R"), g("I_max"), G, g("cells"), g("R0"), g("R1", 0.0), g("C1", 0.0),
                       g("capacity"), g("throttle", 1.0)]
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
    dt = run.get("dt", "auto")
    # explicit-step share: 2% of the stability limit for contacts and springs (impact accuracy); beams are smooth
    # structural elements and run at 20% of theirs (their high modes need stability, not resolution)
    cb = getattr(M, "dt_crit_beam", math.inf)
    other = getattr(M, "dt_crit_other", None)
    M.dt_auto = min(0.02 * M.dt_crit if not math.isfinite(cb) or M.dt_crit < cb else
                    min(0.02 * (other if other is not None else math.inf), 0.2 * cb), M.t_end / 2000.0)
    if getattr(M, "dim", "1d") == "3d":
        from .model3d import rotation_dt3d
        M.dt_auto = min(M.dt_auto, rotation_dt3d(M))
    M.timeout = parse_timeout(run.get("timeout", "auto"), ctx)
    if dt == "auto":
        # automatic step, optionally held inside user bounds (run.dt_min / run.dt_max)
        M.dt = M.dt_auto
        if run.get("dt_max") not in (None, ""):
            M.dt = min(M.dt, num(resolve(run["dt_max"], ctx, "run.dt_max"), "run.dt_max"))
        if run.get("dt_min") not in (None, ""):
            M.dt = max(M.dt, num(resolve(run["dt_min"], ctx, "run.dt_min"), "run.dt_min"))
    else:
        M.dt = num(resolve(dt, ctx, "run.dt"), "run.dt")
    return M


def critical_dt(M):
    if getattr(M, "dim", "1d") == "2d":
        from .model2d import critical_dt2d
        return critical_dt2d(M)
    if getattr(M, "dim", "1d") == "3d":
        from .model3d import critical_dt3d
        return critical_dt3d(M)
    a = M.arrays; best = math.inf
    for e in range(len(M.el_names)):
        t = a["typ"][e]
        k = a["tab_k"][e] if t in (3, 13) else (a["P"][e, 0] if t in (0, 2) else 0.0)
        c = a["C"][e]
        inv = sum(c[i] ** 2 / a["m"][i] for i in range(len(c)) if c[i] != 0.0)
        if t == 8:                       # motor: mechanical time constant J R / (G Kt)^2, stable below 2 tau
            Kt, R, G = a["P"][e, 0], a["P"][e, 1], a["P"][e, 3]
            if inv > 0:
                best = min(best, 2.0 * R / (inv * (G * Kt) ** 2))
            continue
        if k > 0 and inv > 0:
            best = min(best, 2.0 / math.sqrt(k * inv))
    return best if math.isfinite(best) else M.t_end / 100.0


# ------------------------------------------------------------------------------------------- run
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
        if "body" in o and getattr(M, "dim", "1d") == "3d":
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
            qn = o["quantity"]; i = el_index(M, o["element"], qn)
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


def run_model(M, dt=None, history=False, n_log=800, timeout="model", _n=None):
    """timeout: 'model' uses run.timeout; None runs to the end; a number is seconds of wall-clock time."""
    import threading
    a = M.arrays; dt = dt or M.dt
    n = int(round(M.t_end / dt)) if _n is None else _n
    cs, ci, ct, cd, ckeys = _crossings(M)
    every = max(n // n_log, 1) if history else 0
    limit = getattr(M, "timeout", "auto") if timeout == "model" else timeout
    if limit == "auto":
        limit = auto_timeout(M, dt)
    stop = np.zeros(2, dtype=np.int64)
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
    try:
        if dim3:
            from . import engine3d as E3
            (FB, quat, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, tcr, LT, LB, LE, Fe, se, Wg, Lw) = E3.run_engine3d(
                n, dt, a["X0"], a["V0"], a["quat0"], a["Lw0"], a["wsp0"], a["th0"], a["minv"], a["rmode"], a["Iinv"],
                a["axis"], a["Iax"], a["mass"], a["com"], a["grav"], a["typ"], a["one_sided"], a["active"], a["mk"],
                a["ba"], a["bb"], a["ref"], a["P"], a["C"], a["s0"], a["GA"], a["GB"], a["GX"],
                a["tab_off"], a["tab_len"], a["tab_k"], a["tab_x"], a["tab_F"], every, cs, ci, ct, cd, stop)
        elif dim2:
            from . import engine2d as E2
            (q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, SP, aux, tcr, LT, LB, LE, Fe, se, Wg) = E2.run_engine2d(
                n, dt, a["q0"], a["v0"], a["minv"], a["mass"], a["com"], a["grav"], a["typ"], a["one_sided"],
                a["active"], a["mk"], a["ba"], a["bb"], a["ref"], a["P"], a["C"], a["s0"], a["GA"], a["GB"], a["GX"],
                a["tab_off"], a["tab_len"], a["tab_k"], a["tab_x"], a["tab_F"], every, cs, ci, ct, cd, stop)
        else:
            (q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, tcr, LT, LB, LE, Fe, se) = E.run_engine(
                n, dt, a["q0"], a["v0"], a["m"], a["typ"], a["one_sided"], a["active"], a["P"], a["C"], a["s0"],
                a["tab_off"], a["tab_len"], a["tab_k"], a["tab_x"], a["tab_F"], every, cs, ci, ct, cd, stop)
    finally:
        if timer is not None:
            timer.cancel()
    timed_out = int(stop[1]) < n
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
    scale = max(KE0, KE1, inputs, abs(Wg), np.abs(A).sum(), hits, 1e-30)
    led = (abs(glob) + internal) / scale
    R = dict(_ledger_rel=float(led))
    if dim2 or dim3:
        R["_gravity_work"] = float(Wg)
    if timed_out:
        R["_timed_out"] = 1.0
        R["_t_stopped"] = float(stop[1]) * dt
        R["_timeout_s"] = float(limit)
    outs = M.doc["outputs"]
    tc = dict(zip(ckeys, tcr))
    for name, o in outs.items():
        if "expr" in o:
            continue
        red = o.get("reduce", "final")
        if "body" in o and dim3:
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
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]
        elif "body" in o and dim2:
            qn = o["quantity"]
            if qn == "kinetic_energy":
                if red != "final":
                    raise SpecError("kinetic_energy supports reduce: final only", f"outputs.{name}")
                i = M.body_names.index(o["body"]); mi = a["minv"][3 * i:3 * i + 3]
                val = sum(0.5 * v[3 * i + k] ** 2 / mi[k] for k in range(3) if mi[k] > 0)
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
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]
        elif "body" in o:
            i = M.body_names.index(o["body"]); qn = o["quantity"]
            if qn == "kinetic_energy":
                if red != "final":
                    raise SpecError("kinetic_energy supports reduce: final only", f"outputs.{name}")
                val = 0.5 * a["m"][i] * v[i] ** 2
            elif red.startswith("first_time"):
                val = tc[name]
            else:
                lo, hi, fin = (BT[i, 0], BT[i, 1], q[i]) if qn == "position" else (BT[i, 2], BT[i, 3], v[i])
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]
        else:
            qn = o["quantity"]
            grp = (getattr(M, "groups", None) or {}).get(o["element"])
            rows = [M.el_names.index(x) for x in grp.segs] if grp is not None else [M.el_names.index(o["element"])]
            if grp is not None and qn == "measure" and grp.probe:
                rows = [M.el_names.index(grp.probe)]

            def span(e):
                k = {"force": 0, "measure": 1}.get(qn)
                if k is not None:
                    lo, hi, fin = TR[e, 2 * k], TR[e, 2 * k + 1], (Fe[e] if k == 0 else se[e])
                else:
                    j = AUX[qn]
                    lo, hi, fin = TR[e, 4 + 2 * j], TR[e, 5 + 2 * j], aux[e, j]
                return (0.0, 0.0, 0.0) if lo > 1e299 else (lo, hi, fin)
            if red.startswith("first_time"):
                val = tc[name]
            elif qn == "absorbed_work":
                val = sum(A[e] for e in rows)
            elif qn == "dissipated":
                val = sum((D[e] if D[e] >= 0 else A[e] - (S1[e] - S0[e])) for e in rows)
            else:
                sp = [span(e) for e in rows]
                lo = min(x[0] for x in sp); hi = max(x[1] for x in sp); fin = sp[0][2]
                val = {"final": fin, "max": hi, "min": lo, "max_abs": max(abs(lo), abs(hi))}[red]
        val = float(val)
        if red.startswith("first_time"):
            reached = val >= 0
            R[f"_{name}_reached"] = 1.0 if reached else 0.0
            if not reached:
                val = M.t_end * 10.0        # sentinel: never crossed within t_end (flagged in reports)
        if o.get("abs"):
            val = abs(val)
        if "unit" in o:
            val = val / unit_factor(o["unit"])
        R[name] = val
    # expression outputs (may use params and other outputs)
    for name, o in outs.items():
        if "expr" in o:
            nums = numeric_namespace(M.ctx)
            nums.update({k: x for k, x in R.items() if not k.startswith("_")})
            try:
                R[name] = float(eval_numeric(o["expr"], nums))
            except ZeroDivisionError:
                R[name] = math.inf          # e.g. a safety factor when the demand is exactly zero
            except ExprError as exc:
                raise SpecError(str(exc), f"outputs.{name}")
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
        _group_history(M, H)
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
            if M.arrays["typ"][e] == 14:             # rigid impact: running total of the energy it dissipated
                H[f"{en}.dissipated"] = LE[:, e * w + 3]
        _group_history(M, H)
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
        _group_history(M, H)
        return R, H
    return R

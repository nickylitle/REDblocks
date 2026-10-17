"""Control and logic: signals computed every time step from the running model, fed back into it.

A model's `control:` section is a list of named signals. Each step, in the order written, every signal is worked out
from what the model is doing right now (a body's position or speed, an element's force, measure or state, the time)
and from the other signals; elements then read the signals in place of fixed numbers (`k: {control: stiffness}`,
`throttle: {control: thr}`). That closes the loop: air pressure can switch an engine, a speed error can drive a motor.

    control:
      thr_cmd: "= clip(0.4 + 0.002 * (300 - lm.altitude), 0, 1)"        # an expression
      staged:  {if: "air.ambient_pressure < 2000", then: 1, else: 0, latch: true}
      hold:    {pid: {measure: lm.altitude, setpoint: 100, kp: 0.01, ki: 0.001, kd: 0.05, min: 0, max: 1}}
      smooth:  {lag: hold, tau: 0.5 s}
      ramp:    {rate_limit: smooth, rate: 0.2}
      dist:    {integrate: "body.vx", init: 0}

Names in an expression: another signal; `body.quantity` or `element.quantity` (anything an output can read);
`t` (time, s); `$param`; numbers (SI). Functions: min, max, abs, sqrt, exp, log, sin, cos, tan, atan2, clip(x, lo,
hi), sign, floor; comparisons give 1 or 0; `and`, `or`, `not`, `a if cond else b`.

Timing (explicit, like the rest of the engine): at the end of each step the signals read the state that step ended
in (positions, speeds, the forces and element states it worked out) and set the fields the next step uses. The first
step runs on each controlled field's init value. A signal that reads one written below it gets that one's value from
the step before (a one-step delay), so any feedback loop is well defined.
"""
import ast
import math
import re

import numpy as np
from numba import njit

from .loader import SpecError

# signal kinds
K_EXPR, K_IF, K_PID, K_LAG, K_RATE, K_INT = 0, 1, 2, 3, 4, 5
KIND_KEYS = {"if": K_IF, "pid": K_PID, "lag": K_LAG, "rate_limit": K_RATE, "integrate": K_INT}
# program ops
OP_C, OP_S, OP_T, OP_X = 0, 1, 2, 3          # push constant, signal, time, sensor
(OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_NEG, OP_MIN, OP_MAX, OP_ABS, OP_SQRT, OP_EXP, OP_LOG, OP_SIN, OP_COS,
 OP_TAN, OP_ATAN2, OP_GT, OP_GE, OP_LT, OP_LE, OP_EQ, OP_NE, OP_AND, OP_OR, OP_NOT, OP_SEL, OP_CLIP, OP_SIGN,
 OP_FLOOR, OP_MOD) = range(10, 40)
FUNC_OPS = {"min": (OP_MIN, 2), "max": (OP_MAX, 2), "abs": (OP_ABS, 1), "sqrt": (OP_SQRT, 1), "exp": (OP_EXP, 1),
            "log": (OP_LOG, 1), "sin": (OP_SIN, 1), "cos": (OP_COS, 1), "tan": (OP_TAN, 1), "atan2": (OP_ATAN2, 2),
            "clip": (OP_CLIP, 3), "sign": (OP_SIGN, 1), "floor": (OP_FLOOR, 1)}
BIN_OPS = {ast.Add: OP_ADD, ast.Sub: OP_SUB, ast.Mult: OP_MUL, ast.Div: OP_DIV, ast.Pow: OP_POW, ast.Mod: OP_MOD}
CMP_OPS = {ast.Gt: OP_GT, ast.GtE: OP_GE, ast.Lt: OP_LT, ast.LtE: OP_LE, ast.Eq: OP_EQ, ast.NotEq: OP_NE}
NPRM = 10
_PARAM = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
SENTINEL = 7.123456789e-123                    # marks a controlled field while the model compiles


def empty_control():
    """Arrays for a model with no control section."""
    return dict(kind=np.zeros(0, np.int64), prog=np.zeros(1, np.int64), val=np.zeros(1), rng=np.zeros((0, 3, 2), np.int64),
                prm=np.zeros((0, NPRM)), ssrc=np.zeros(0, np.int64), sidx=np.zeros(0, np.int64),
                ae=np.zeros(0, np.int64), aslot=np.zeros(0, np.int64), asig=np.zeros(0, np.int64),
                aarr=np.zeros(0, np.int64), ctlD=np.zeros(1, np.int64))


def control_args(M):
    c = getattr(M, "ctl", None) or empty_control()
    return (c["kind"], c["prog"], c["val"], c["rng"], c["prm"], c["ssrc"], c["sidx"], c["ae"], c["aslot"], c["asig"],
            c["aarr"], c["ctlD"])


# ------------------------------------------------------------------------------------------------ compile
class _Prog:
    def __init__(self):
        self.ops = []; self.vals = []

    def emit(self, op, v=0.0):
        self.ops.append(op); self.vals.append(float(v))


def _subst_params(text, ctx, where):
    from .model import numeric_namespace
    nums = numeric_namespace(ctx)

    def rep(m):
        k = m.group(1)
        if k not in nums:
            raise SpecError(f"undefined parameter ${k}", where)
        return repr(float(nums[k]))
    return _PARAM.sub(rep, text)


def compile_expr(text, ctx, names, sensor, where, prog):
    """Emit an expression into prog. names: signal name -> index. sensor(name, qty) -> sensor index."""
    if isinstance(text, (int, float)):
        prog.emit(OP_C, float(text)); return
    from .model import resolve, num
    s = str(text).strip()
    if s.startswith("="):
        s = s[1:].strip()
    else:
        # a plain value with a unit ("0.5 s"), a $param, or a single name
        try:
            prog.emit(OP_C, num(resolve(s, ctx, where), where)); return
        except Exception:
            pass
    s = _subst_params(s, ctx, where)
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise SpecError(f"cannot read the expression '{text}': {exc.msg}", where)

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            prog.emit(OP_C, n.value); return
        if isinstance(n, ast.Constant) and isinstance(n.value, bool):
            prog.emit(OP_C, 1.0 if n.value else 0.0); return
        if isinstance(n, ast.Name):
            if n.id in ("t", "time"):
                prog.emit(OP_T); return
            if n.id in ("pi",):
                prog.emit(OP_C, math.pi); return
            if n.id in names:
                prog.emit(OP_S, names[n.id]); return
            raise SpecError(f"'{n.id}' is not a control signal (signals: {', '.join(names) or 'none'}); to read the "
                            "model use body.quantity or element.quantity", where)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            prog.emit(OP_X, sensor(n.value.id, n.attr, where)); return
        if isinstance(n, ast.BinOp) and type(n.op) in BIN_OPS:
            walk(n.left); walk(n.right); prog.emit(BIN_OPS[type(n.op)]); return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            walk(n.operand); prog.emit(OP_NEG); return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            walk(n.operand); return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            walk(n.operand); prog.emit(OP_NOT); return
        if isinstance(n, ast.Compare):
            if len(n.ops) != 1 or type(n.ops[0]) not in CMP_OPS:
                if len(n.ops) == 2:                                   # a < x < b
                    walk(ast.BoolOp(op=ast.And(), values=[ast.Compare(left=n.left, ops=[n.ops[0]], comparators=[n.comparators[0]]),
                                                          ast.Compare(left=n.comparators[0], ops=[n.ops[1]], comparators=[n.comparators[1]])]))
                    return
                raise SpecError("comparisons are <, <=, >, >=, ==, !=", where)
            walk(n.left); walk(n.comparators[0]); prog.emit(CMP_OPS[type(n.ops[0])]); return
        if isinstance(n, ast.BoolOp):
            op = OP_AND if isinstance(n.op, ast.And) else OP_OR
            walk(n.values[0])
            for v in n.values[1:]:
                walk(v); prog.emit(op)
            return
        if isinstance(n, ast.IfExp):
            walk(n.test); walk(n.body); walk(n.orelse); prog.emit(OP_SEL); return
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNC_OPS:
            op, na = FUNC_OPS[n.func.id]
            if len(n.args) != na:
                raise SpecError(f"{n.func.id} takes {na} argument{'s' if na > 1 else ''}", where)
            for a in n.args:
                walk(a)
            prog.emit(op); return
        raise SpecError(f"'{ast.unparse(n)}' is not allowed in a control expression", where)
    walk(tree)


def compile_control(doc, ctx, M, sensor_slot):
    """sensor_slot(name, qty, where) -> (src, idx) for this model's engine."""
    from .model import resolve, num
    sec = doc.get("control") or {}
    if not sec:
        return None
    if not isinstance(sec, dict):
        raise SpecError("control is a mapping of signal names to expressions or blocks", "control")
    names = {k: i for i, k in enumerate(sec)}
    for k in names:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            raise SpecError(f"control signal '{k}' needs a plain name (letters, digits, _)", f"control.{k}")
        if k in ("t", "time"):
            raise SpecError("'t' and 'time' are the clock: pick another signal name", f"control.{k}")
    sens_keys = []; ssrc = []; sidx = []

    def sensor(name, qty, where):
        key = (name, qty)
        if key not in sens_keys:
            s, i = sensor_slot(name, qty, where)
            sens_keys.append(key); ssrc.append(s); sidx.append(i)
        return sens_keys.index(key)
    ns = len(sec)
    kind = np.zeros(ns, np.int64); rng = np.zeros((ns, 3, 2), np.int64); prm = np.zeros((ns, NPRM))
    prog = _Prog()

    def inp(j, slot, text, where):
        a = len(prog.ops)
        compile_expr(text, ctx, names, sensor, where, prog)
        rng[j, slot] = (a, len(prog.ops))
    g = lambda d, k, where, dflt: num(resolve(d[k], ctx, f"{where}.{k}"), f"{where}.{k}") if k in d else dflt
    for name, spec in sec.items():
        j = names[name]; where = f"control.{name}"
        if not isinstance(spec, dict):
            kind[j] = K_EXPR; inp(j, 0, spec, where); continue
        ks = [k for k in spec if k in KIND_KEYS] + (["expr"] if "expr" in spec else [])
        if len(ks) != 1:
            raise SpecError("a control signal is an expression, or one of if, pid, lag, rate_limit, integrate, expr",
                            where)
        k = ks[0]
        extra = set(spec) - {k, "then", "else", "latch", "for", "tau", "rate", "init", "label", "note"}
        if extra:
            raise SpecError(f"unknown key{'s' if len(extra) > 1 else ''} {', '.join(sorted(extra))} in a {k} signal", where)
        if k == "expr":
            kind[j] = K_EXPR; inp(j, 0, spec["expr"], where)
        elif k == "if":
            kind[j] = K_IF
            inp(j, 0, spec["if"], where + ".if"); inp(j, 1, spec.get("then", 1), where + ".then")
            inp(j, 2, spec.get("else", 0), where + ".else")
            prm[j, 0] = 1.0 if spec.get("latch") else 0.0
            prm[j, 1] = g(spec, "for", where, 0.0)
        elif k == "pid":
            p = spec["pid"]
            if not isinstance(p, dict) or "measure" not in p or "setpoint" not in p:
                raise SpecError("pid: {measure, setpoint, kp, ki, kd, min, max, d_filter}", where)
            bad = set(p) - {"measure", "setpoint", "kp", "ki", "kd", "min", "max", "d_filter", "bias"}
            if bad:
                raise SpecError(f"unknown pid key{'s' if len(bad) > 1 else ''} {', '.join(sorted(bad))}", where)
            kind[j] = K_PID
            inp(j, 0, p["measure"], where + ".measure"); inp(j, 1, p["setpoint"], where + ".setpoint")
            gp = lambda key, dflt: num(resolve(p[key], ctx, f"{where}.pid.{key}"), f"{where}.pid.{key}") if key in p else dflt
            prm[j, :7] = [gp("kp", 0.0), gp("ki", 0.0), gp("kd", 0.0), gp("min", -1e300), gp("max", 1e300),
                          gp("d_filter", 0.0), gp("bias", 0.0)]
            if prm[j, 3] > prm[j, 4]:
                raise SpecError("pid min is above max", where)
        elif k == "lag":
            kind[j] = K_LAG; inp(j, 0, spec["lag"], where + ".lag"); prm[j, 0] = g(spec, "tau", where, 0.0)
            prm[j, 1] = g(spec, "init", where, math.nan)
        elif k == "rate_limit":
            kind[j] = K_RATE; inp(j, 0, spec["rate_limit"], where + ".rate_limit")
            prm[j, 0] = g(spec, "rate", where, None) if "rate" in spec else None
            if prm[j, 0] is None or not prm[j, 0] > 0:
                raise SpecError("rate_limit needs rate: the most it may change per second (> 0)", where)
            prm[j, 1] = g(spec, "init", where, math.nan)
        elif k == "integrate":
            kind[j] = K_INT; inp(j, 0, spec["integrate"], where + ".integrate"); prm[j, 0] = g(spec, "init", where, 0.0)
    return dict(names=list(sec), sensors=sens_keys, kind=kind, prog=np.array(prog.ops or [0], np.int64),
                val=np.array(prog.vals or [0.0]), rng=rng, prm=prm, ssrc=np.array(ssrc, np.int64),
                sidx=np.array(sidx, np.int64))


def controlled_fields(doc):
    """Every field written {control: signal, init: ..., max: ...}: in an element, or in a circuit part (including
    its esc and thermal settings). Returns [dict(path, where, signal, init, max)]."""
    out = []

    def take(path, where, v):
        bad = set(v) - {"control", "init", "max"}
        if bad:
            raise SpecError(f"{path[-1]}: {{control: <signal>, init: <first-step value>, max: <the most it reaches>}}", where)
        out.append(dict(path=path, where=where, signal=str(v["control"]), init=v.get("init", 0.0), max=v.get("max")))
    for en, d in (doc.get("elements") or {}).items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "control" in v:
                    take(("elements", en, k), f"elements.{en}.{k}", v)
    parts = ((doc.get("circuit") or {}).get("parts") or {}) if isinstance(doc.get("circuit"), dict) else {}
    for pn, d in parts.items():
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if isinstance(v, dict) and "control" in v:
                take(("circuit", "parts", pn, k), f"circuit.parts.{pn}.{k}", v)
            elif isinstance(v, dict) and k in ("esc", "thermal"):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and "control" in vv:
                        take(("circuit", "parts", pn, k, kk), f"circuit.parts.{pn}.{k}.{kk}", vv)
    return out


def with_values(doc, fields, values):
    """A copy of doc whose controlled fields hold the given numbers (only the branches on the way are copied)."""
    import copy
    d2 = copy.copy(doc)
    if hasattr(doc, "lines"):
        d2.lines = doc.lines
    for fd, v in zip(fields, values):
        node = d2
        for key in fd["path"][:-1]:
            child = dict(node[key]); node[key] = child; node = child
        node[fd["path"][-1]] = v
    return d2


# ------------------------------------------------------------------------------------------------ run
@njit(cache=True, nogil=True)
def _run_prog(prog, val, a, b, sig, sens, t, stack):
    sp = 0
    for k in range(a, b):
        op = prog[k]
        if op == OP_C:
            stack[sp] = val[k]; sp += 1
        elif op == OP_S:
            stack[sp] = sig[int(val[k])]; sp += 1
        elif op == OP_T:
            stack[sp] = t; sp += 1
        elif op == OP_X:
            stack[sp] = sens[int(val[k])]; sp += 1
        elif op == OP_NEG or op == OP_ABS or op == OP_SQRT or op == OP_EXP or op == OP_LOG or op == OP_SIN or \
                op == OP_COS or op == OP_TAN or op == OP_NOT or op == OP_SIGN or op == OP_FLOOR:
            x = stack[sp - 1]
            if op == OP_NEG:
                r = -x
            elif op == OP_ABS:
                r = abs(x)
            elif op == OP_SQRT:
                r = math.sqrt(x) if x > 0.0 else 0.0
            elif op == OP_EXP:
                r = math.exp(min(x, 700.0))
            elif op == OP_LOG:
                r = math.log(x) if x > 0.0 else -1e300
            elif op == OP_SIN:
                r = math.sin(x)
            elif op == OP_COS:
                r = math.cos(x)
            elif op == OP_TAN:
                r = math.tan(x)
            elif op == OP_NOT:
                r = 1.0 if x == 0.0 else 0.0
            elif op == OP_SIGN:
                r = 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)
            else:
                r = math.floor(x)
            stack[sp - 1] = r
        elif op == OP_SEL or op == OP_CLIP:                        # three operands
            c = stack[sp - 3]; x = stack[sp - 2]; y = stack[sp - 1]; sp -= 2
            if op == OP_SEL:
                stack[sp - 1] = x if c != 0.0 else y
            else:
                stack[sp - 1] = min(max(c, x), y)
        else:
            x = stack[sp - 2]; y = stack[sp - 1]; sp -= 1
            if op == OP_ADD:
                r = x + y
            elif op == OP_SUB:
                r = x - y
            elif op == OP_MUL:
                r = x * y
            elif op == OP_DIV:
                r = x / y if y != 0.0 else (1e300 if x > 0.0 else (-1e300 if x < 0.0 else 0.0))
            elif op == OP_POW:
                r = x ** y if (x > 0.0 or y == math.floor(y)) else 0.0
            elif op == OP_MIN:
                r = min(x, y)
            elif op == OP_MAX:
                r = max(x, y)
            elif op == OP_ATAN2:
                r = math.atan2(x, y)
            elif op == OP_GT:
                r = 1.0 if x > y else 0.0
            elif op == OP_GE:
                r = 1.0 if x >= y else 0.0
            elif op == OP_LT:
                r = 1.0 if x < y else 0.0
            elif op == OP_LE:
                r = 1.0 if x <= y else 0.0
            elif op == OP_EQ:
                r = 1.0 if x == y else 0.0
            elif op == OP_NE:
                r = 1.0 if x != y else 0.0
            elif op == OP_AND:
                r = 1.0 if (x != 0.0 and y != 0.0) else 0.0
            elif op == OP_OR:
                r = 1.0 if (x != 0.0 or y != 0.0) else 0.0
            elif op == OP_MOD:
                r = x - y * math.floor(x / y) if y != 0.0 else 0.0
            else:
                r = 0.0
            stack[sp - 1] = r
    return stack[0] if sp > 0 else 0.0


@njit(cache=True, nogil=True)
def control_step(t, dt, first, kind, prog, val, rng, prm, cst, sig, sens, stack):
    """Every signal, in order. cst[j]: [state 0..3]. first: the first step (initialise lags and limiters)."""
    for j in range(kind.shape[0]):
        k = kind[j]
        u = _run_prog(prog, val, rng[j, 0, 0], rng[j, 0, 1], sig, sens, t, stack)
        if k == 0:
            sig[j] = u
        elif k == 1:                                   # if / else, with latch and a hold time
            on = u != 0.0
            if prm[j, 1] > 0.0:                        # the condition must hold this long first
                if on:
                    cst[j, 1] += dt
                else:
                    cst[j, 1] = 0.0
                on = cst[j, 1] >= prm[j, 1] - 1e-12
            if prm[j, 0] > 0.5:
                if on:
                    cst[j, 0] = 1.0
                on = cst[j, 0] > 0.5
            if on:
                sig[j] = _run_prog(prog, val, rng[j, 1, 0], rng[j, 1, 1], sig, sens, t, stack)
            else:
                sig[j] = _run_prog(prog, val, rng[j, 2, 0], rng[j, 2, 1], sig, sens, t, stack)
        elif k == 2:                                   # PID on the error setpoint - measure
            sp_ = _run_prog(prog, val, rng[j, 1, 0], rng[j, 1, 1], sig, sens, t, stack)
            e = sp_ - u
            kp = prm[j, 0]; ki = prm[j, 1]; kd = prm[j, 2]; lo = prm[j, 3]; hi = prm[j, 4]; tf = prm[j, 5]
            if first:
                cst[j, 1] = u; cst[j, 2] = 0.0         # previous measure, filtered derivative
            dm = (u - cst[j, 1]) / dt if not first else 0.0
            if tf > 0.0:
                a = dt / (tf + dt)
                cst[j, 2] += a * (dm - cst[j, 2])
            else:
                cst[j, 2] = dm
            cst[j, 1] = u
            # derivative on the measure (no kick when the setpoint steps)
            out = prm[j, 6] + kp * e + cst[j, 0] - kd * cst[j, 2]
            trial = cst[j, 0] + ki * e * dt
            un = prm[j, 6] + kp * e + trial - kd * cst[j, 2]
            # integrate only while that does not push further into a limit (anti-windup)
            if not ((un > hi and e * ki > 0.0) or (un < lo and e * ki < 0.0)):
                cst[j, 0] = trial
                out = un
            sig[j] = min(max(out, lo), hi)
        elif k == 3:                                   # first-order lag
            if first:
                cst[j, 0] = prm[j, 1] if prm[j, 1] == prm[j, 1] else u
            tau = prm[j, 0]
            if tau > 0.0:
                cst[j, 0] += (u - cst[j, 0]) * (1.0 - math.exp(-dt / tau))
            else:
                cst[j, 0] = u
            sig[j] = cst[j, 0]
        elif k == 4:                                   # rate limiter
            if first:
                cst[j, 0] = prm[j, 1] if prm[j, 1] == prm[j, 1] else u
            d = u - cst[j, 0]; m = prm[j, 0] * dt
            if d > m:
                d = m
            elif d < -m:
                d = -m
            cst[j, 0] += d
            sig[j] = cst[j, 0]
        elif k == 5:                                   # integrator
            if first:
                cst[j, 0] = prm[j, 0]
            cst[j, 0] += u * dt
            sig[j] = cst[j, 0]

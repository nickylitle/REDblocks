"""Optimizer: several values varied at once to minimize (or maximize) one output, keeping chosen outputs inside
limits and, optionally, every failure mode passing. Each trial is one ordinary run of the model.

    study:
      optimize:
        vary: {t_plate: [1 mm, 6 mm], w_rib: [5 mm, 30 mm]}     # each value with its range
        minimize: mass                                           # or maximize: an output
        subject_to:                                              # in each output's own unit
          - {output: peak_force, max: 5 kN}
          - {output: sf_bend, min: 2}
        failure_modes: true                                      # every failure mode passes (SF >= 1)
        max_runs: 150

Method: COBYLA (constrained optimization by linear approximation, Powell 1994): it needs no derivatives, so steps
in an output (a contact that starts, a part that yields) do not stall it. Each value is scaled to 0..1 over its range
and kept inside it. A trial that errors counts as infeasible. The answer is the best feasible trial seen."""
import math

import numpy as np

from .loader import SpecError
from .units import parse_quantity, unit_factor


def spec(doc, s):
    g = (doc.get("study") or {}).get("optimize")
    if not g:
        return None
    where = "study.optimize"
    vary = g.get("vary") or {}
    if not isinstance(vary, dict) or not vary:
        raise SpecError("optimize.vary lists each value with its range: {t: [1 mm, 6 mm]}", where)
    ctx = s.nominal.ctx; V = []
    for k, r in vary.items():
        if not isinstance(ctx.get(k), (int, float)) or isinstance(ctx.get(k), bool):
            raise SpecError(f"optimize.vary: '{k}' is not a number value of this model", where)
        if k not in (doc.get("params") or {}):
            raise SpecError(f"optimize.vary: '{k}' is worked out from other values (derived), so it cannot be varied; "
                            "vary the params it comes from", where)
        if not isinstance(r, list) or len(r) != 2:
            raise SpecError(f"optimize.vary.{k}: give a range [low, high]", where)
        lo, hi = (float(parse_quantity(x)) if isinstance(x, str) else float(x) for x in r)
        if not hi > lo:
            raise SpecError(f"optimize.vary.{k}: high must be above low", where)
        V.append((k, lo, hi, float(ctx[k])))
    outs = doc["outputs"]
    if ("minimize" in g) == ("maximize" in g):
        raise SpecError("optimize needs one of minimize: or maximize: (an output)", where)
    obj = str(g.get("minimize") or g.get("maximize")); sign = 1.0 if "minimize" in g else -1.0
    if obj not in outs:
        raise SpecError(f"optimize: no output named '{obj}'", where)
    cons = []
    for i, c in enumerate(g.get("subject_to") or []):
        o = str(c.get("output", ""))
        if o not in outs:
            raise SpecError(f"optimize.subject_to[{i}]: no output named '{o}'", where)
        u = outs[o].get("unit", "")
        for side in ("min", "max"):
            if side in c:
                val = c[side]
                val = float(parse_quantity(val)) / (unit_factor(u) if u else 1.0) if isinstance(val, str) else float(val)
                cons.append((o, side, val))
    if not cons and not g.get("failure_modes"):
        pass                                    # unconstrained: fine, the ranges still bound it
    return dict(vary=V, objective=obj, sign=sign, cons=cons, failure_modes=bool(g.get("failure_modes")),
                max_runs=int(g.get("max_runs", 150)))


def optimize(s, G, cancel=lambda: False, log=None):
    """Returns dict(found, values, objective, outputs, runs, reason)."""
    from scipy.optimize import minimize
    V = G["vary"]; n = len(V)
    fms = s.failure_modes() if G["failure_modes"] else []
    cache = {}; runs = []

    def to_x(u):
        return {k: lo + min(max(ui, 0.0), 1.0) * (hi - lo) for (k, lo, hi, _), ui in zip(V, u)}

    def ev(u):
        key = tuple(round(float(x), 9) for x in u)
        if key in cache:
            return cache[key]
        if cancel():
            raise KeyboardInterrupt
        x = to_x(u)
        try:
            o = s.run(x)
            f = G["sign"] * float(o[G["objective"]])
            g = []
            for name, side, val in G["cons"]:
                y = float(o[name]); sc = max(abs(val), 1e-12)
                g.append(((y - val) if side == "min" else (val - y)) / sc)
            for fm in fms:
                failed, sf = fm.evaluate(o)
                g.append(min(sf, 10.0) - 1.0 if math.isfinite(sf) else 9.0)
            ok = all(v >= -1e-9 for v in g)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            o = {"_error": str(exc)}; f = math.inf; g = [-1.0] * (len(G["cons"]) + len(fms)); ok = False
        r = dict(values=x, objective=(G["sign"] * f if math.isfinite(f) else None), feasible=ok,
                 outputs={k: v for k, v in o.items() if not k.startswith("_")}, error=o.get("_error"))
        runs.append(r)
        if log is not None:
            log.append(r)
        cache[key] = (f, g)
        return cache[key]

    u0 = np.array([(x0 - lo) / (hi - lo) for _, lo, hi, x0 in V]).clip(0, 1)
    ncon = len(G["cons"]) + len(fms)
    cons = [{"type": "ineq", "fun": (lambda u, i=i: ev(u)[1][i])} for i in range(ncon)]
    cons += [{"type": "ineq", "fun": (lambda u, j=j: u[j])} for j in range(n)]
    cons += [{"type": "ineq", "fun": (lambda u, j=j: 1.0 - u[j])} for j in range(n)]
    reason = ""
    try:
        res = minimize(lambda u: ev(u)[0] if math.isfinite(ev(u)[0]) else 1e300, u0, method="COBYLA",
                       constraints=cons, options=dict(maxiter=G["max_runs"], rhobeg=0.25, tol=1e-4))
        reason = str(res.message)
    except KeyboardInterrupt:
        reason = "cancelled"
    feas = [r for r in runs if r["feasible"] and r["objective"] is not None]
    if not feas:
        best = min((r for r in runs if r["objective"] is not None), default=None,
                   key=lambda r: G["sign"] * r["objective"])
        return dict(found=False, best=best, runs=runs, reason="no trial met every limit" + (f" ({reason})" if reason else ""),
                    spec=_public(G))
    best = min(feas, key=lambda r: G["sign"] * r["objective"])
    return dict(found=True, best=best, runs=runs, reason=reason, spec=_public(G))


def _public(G):
    return dict(vary=[dict(name=k, lo=lo, hi=hi, start=x0) for k, lo, hi, x0 in G["vary"]], objective=G["objective"],
                goal="minimize" if G["sign"] > 0 else "maximize",
                cons=[dict(output=o, side=s_, value=v) for o, s_, v in G["cons"]], failure_modes=G["failure_modes"])

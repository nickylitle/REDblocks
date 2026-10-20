"""Parameter estimation: find the parameter values that make a model's time history match measured test data
(Simulink Design Optimization's parameter estimation, for REDblocks).

    from simblocks.diagnostics.estimate import estimate
    r = estimate(load_system("drive.yaml"), data="launch_test.csv", signal="chassis.vx",
                 vary={"mu_tire": ["1.0", "2.5"], "k_tire": ["5 kN/m", "50 kN/m"]})

- data: a CSV (first column time, the next the measurement; or time_col / y_col by name or number, with time_unit
  and unit for its units) or a dict {t: [...], y: [...]} in SI.
- signal: what to compare, as `body.quantity`, `element.quantity`, `part.quantity` or `control.name`, or a column
  spec like the exports' ({body: chassis, quantity: ax, unit: gn}). Body accelerations (acceleration, ax, ay, az)
  are the derivative of the logged velocity.
- vary: {param: [low, high]} (numbers in SI, or quantities with units). The search keeps inside the bounds; a range
  spanning more than 20x is searched on a log scale.

The search is bounded least squares (scipy's trust-region reflective) on the misfit at the measurement times; each
evaluation is one full run. The result has the best values, their 1-sigma uncertainty from the fit's Jacobian
(how well the data pins each one down), the correlation between them, the RMS misfit and R^2, and the fitted curve.
A parameter the data cannot see shows a huge uncertainty: that is the honest answer, not a failure.
"""
import math
import os

import numpy as np


def _si(v):
    if isinstance(v, (int, float)):
        return float(v)
    from ..compiler.exports import unit_factor_value
    return unit_factor_value(v)


def read_data(data, time_col=0, y_col=1, time_unit="s", unit=None):
    """(t in s, y in SI) from a CSV path or a {t, y} dict."""
    if isinstance(data, dict):
        return np.asarray(data["t"], float), np.asarray(data["y"], float)
    from ..compiler.units import unit_factor
    with open(os.path.expanduser(str(data)), encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    head = None
    try:
        float(lines[0].split(",")[0])
    except ValueError:
        head = [h.strip() for h in lines[0].split(",")]; lines = lines[1:]
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines])
    col = lambda c: head.index(c) if isinstance(c, str) and head and c in head else int(c)
    t = rows[:, col(time_col)] * unit_factor(time_unit or "s")
    y = rows[:, col(y_col)] * (unit_factor(unit) if unit else 1.0)
    return t, y


def _spec(signal):
    if isinstance(signal, dict):
        return dict(signal)
    name, q = str(signal).split(".", 1)
    return {"__key__": name, "quantity": q}


def _series(system, M, H, spec):
    from ..compiler.exports import column
    c = dict(spec)
    if "__key__" in c:
        name = c.pop("__key__")
        doc = system.doc
        if name == "control":
            c = {"control": c["quantity"]}
        elif name in (doc.get("bodies") or {}):
            c["body"] = name
        elif name in ((doc.get("circuit") or {}).get("parts") or {}):
            c["part"] = name
        else:
            c["element"] = name
    c.pop("unit", None)                               # compare in SI
    y, _, _ = column(system.doc, H, c, M)
    return np.asarray(H["t"], float), y


def estimate(system, data, signal, vary, t_range=None, x0=None, max_runs=300, time_col=0, y_col=1, time_unit="s",
             unit=None, log=None):
    td, yd = read_data(data, time_col, y_col, time_unit, unit)
    if t_range:
        m = (td >= _si(t_range[0])) & (td <= _si(t_range[1])); td, yd = td[m], yd[m]
    if len(td) < 3:
        raise ValueError("estimation needs at least 3 data points in the time range")
    names = list(vary)
    lo = np.array([_si(vary[k][0]) for k in names]); hi = np.array([_si(vary[k][1]) for k in names])
    if (hi <= lo).any():
        raise ValueError("each vary range is [low, high] with low < high")
    logs = (lo > 0) & (hi / np.where(lo > 0, lo, 1) > 20)
    to_p = lambda u: np.where(logs, lo * (hi / np.where(lo > 0, lo, 1)) ** u, lo + (hi - lo) * u)
    to_u = lambda p: np.where(logs, np.log(p / lo) / np.log(np.where(logs, hi / np.where(lo > 0, lo, 1), 2.0)), (p - lo) / (hi - lo))
    spec = _spec(signal)
    T_end = float(td.max())
    runs = {"n": 0}

    def model(p):
        ov = {k: float(v) for k, v in zip(names, p)}
        M = system.compile(ov)
        from ..compiler.model import run_model
        n_log = int(min(max(4 * len(td), 2000), M.t_end / M.dt + 1))
        r = run_model(M, history=True, n_log=n_log)
        from ..compiler.system import _with_blocks
        _, H = _with_blocks(M, r, True)
        t, y = _series(system, M, H, spec)
        runs["n"] += 1
        if t[-1] < T_end - max(1e-9 * T_end, 2.0 * M.dt):
            raise ValueError(f"the model ends at t = {t[-1]:.4g} s, before the data does ({T_end:.4g} s): raise run.t_end")
        return np.interp(td, t, y)

    scale = max(float(np.std(yd)), float(np.max(np.abs(yd))) * 1e-3, 1e-30)

    def resid(u):
        r = (model(to_p(u)) - yd) / scale
        if log is not None:
            log.append(dict(values={k: float(v) for k, v in zip(names, to_p(u))}, rms=float(np.sqrt(np.mean(r * r))) * scale))
        return r

    from scipy.optimize import least_squares
    if x0:
        u0 = to_u(np.array([_si(x0[k]) for k in names]))
    else:
        # a fit to an oscillation has false minima (a period off by one cycle): start from the best of a spread of
        # trial points (the centre, a grid along each axis and a few space-filling ones)
        n = len(names); rng = np.random.default_rng(12345)
        cand = [np.full(n, 0.5)] + [np.eye(n)[i] * f + (1 - np.eye(n)[i]) * 0.5 for i in range(n) for f in (0.1, 0.3, 0.7, 0.9)]
        cand += list((rng.permuted(np.tile(np.arange(8), (n, 1)), axis=1).T + rng.random((8, n))) / 8)
        best = None
        for u in cand:
            try:
                v = float(np.sum(((model(to_p(np.clip(u, 0, 1))) - yd) / scale) ** 2))
            except ValueError:
                raise
            if best is None or v < best[0]:
                best = (v, np.clip(u, 0, 1))
        u0 = best[1]
    sol = least_squares(resid, np.clip(u0, 0, 1), bounds=(np.zeros(len(names)), np.ones(len(names))), method="trf",
                        x_scale=1.0, diff_step=1e-4, max_nfev=max(10, max_runs // (len(names) + 1)))
    p = to_p(sol.x)
    fit = model(p)
    res = fit - yd; dof = max(1, len(td) - len(names))
    s2 = float(np.sum(res ** 2)) / dof
    J = sol.jac * scale                                 # d(misfit)/du in SI
    dpdu = np.where(logs, p * np.log(np.where(logs, hi / np.where(lo > 0, lo, 1), 2.0)), hi - lo)
    try:
        cov_u = np.linalg.pinv(J.T @ J) * s2
        cov = cov_u * np.outer(dpdu, dpdu)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None)); corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        std = np.full(len(names), np.inf); corr = np.eye(len(names))
    at_bound = [names[i] for i in range(len(names)) if sol.x[i] < 1e-6 or sol.x[i] > 1 - 1e-6]
    ss_tot = float(np.sum((yd - yd.mean()) ** 2))
    return dict(values={k: float(v) for k, v in zip(names, p)}, std={k: float(s) for k, s in zip(names, std)},
                correlation={a: {b: float(corr[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
                rms=float(math.sqrt(np.mean(res ** 2))), r2=float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
                runs=runs["n"], at_bound=at_bound, success=bool(sol.success), message=str(sol.message),
                t=td.tolist(), measured=yd.tolist(), fitted=fit.tolist())

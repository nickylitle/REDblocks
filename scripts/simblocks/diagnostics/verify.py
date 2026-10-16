"""Automated verification battery that EVERY assembled sim runs before its results are reported.

A sim is wrapped in a SimSpec:  run(params, dt) -> {metric: float, ...}
Reserved metric names the harness understands (optional, the sim reports them if it can):
    _ledger_rel        energy-ledger closure residual / scale        (checked against ledger_tol)
    _momentum_rel      linear-momentum drift / scale                  (checked if listed in `conserved`)
    _angmom_rel        angular-momentum drift / scale                 (idem)
Any other underscore metric is treated as a diagnostic and not convergence-checked.

verify_all(spec) runs, in order:
  determinism, NaN/inf guard, physical bounds, stability margin, timestep convergence (dt, dt/2, dt/4,
  observed order, Richardson), energy ledger at every dt, conservation laws, limit cases, knockouts,
  one-at-a-time sensitivity, Monte Carlo (percentiles + rank correlations + failure rate), anchors.
Monotonicity is deliberately NOT checked: real behavior can be non-monotonic, and an expectation baked
into the harness would flag correct physics as a failure. Trends are reported by the sensitivity sweep only.
"""
import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..registry import block

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class SimSpec:
    name: str
    run: Callable                      # run(params: dict, dt: float) -> dict of floats
    params: dict                       # nominal parameters
    dt: float                          # step used for reported results
    metrics: list = None               # metrics to convergence-check (default: all non-underscore)
    uncertain: dict = field(default_factory=dict)    # name -> (lo, hi) uniform, or callable(rng) -> value
    knockouts: dict = field(default_factory=dict)    # name -> overrides  or  (overrides, {metric: '+'|'-'|'any'})
    limits: list = field(default_factory=list)       # (description, overrides, {metric: value | callable(v)->bool})
    bounds: dict = field(default_factory=dict)       # metric -> (lo, hi)   physical admissibility
    critical_dt: Callable = None       # critical_dt(params) -> float
    anchors: dict = field(default_factory=dict)      # metric -> (value, rel_tol, source)
    conserved: list = field(default_factory=list)    # reserved metric names that must stay ~0
    conserved_tol: float = 1e-6
    ledger_tol: float = 0.05
    conv_tol: float = 0.01
    conv_levels: int = 3
    mc_n: int = 200
    mc_seed: int = 7
    knockout_dead_tol: float = 1e-9
    dt_accuracy_margin: float = 5.0    # warn if critical_dt / dt below this
    failure_modes: list = field(default_factory=list)  # FailureMode list (user-defined; see failure.py)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    data: dict = field(default_factory=dict)


class Report:
    def __init__(self, spec):
        self.spec = spec; self.checks = []; self.t0 = time.time(); self.runs = 0

    def add(self, *a, **k):
        self.checks.append(Check(*a, **k))

    @property
    def ok(self):
        return all(c.status != FAIL for c in self.checks)

    def summary(self):
        n = {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, FAIL, SKIP)}
        return f"{self.spec.name}: {'VERIFIED' if self.ok else 'NOT VERIFIED'} | " + \
            " ".join(f"{k} {v}" for k, v in n.items()) + f" | {self.runs} runs, {time.time() - self.t0:.1f} s"

    def markdown(self):
        lines = [f"# Verification report: {self.spec.name}", "", f"**{self.summary()}**", "",
                 "| check | status | detail |", "|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.name} | {c.status} | {c.detail} |")
        mc = next((c for c in self.checks if c.name == "monte_carlo"), None)
        if mc and mc.data.get("percentiles"):
            lines += ["", "## Monte Carlo", "", "| metric | P5 | P50 | P95 | top driver (rank corr) |", "|---|---|---|---|---|"]
            for m, q in mc.data["percentiles"].items():
                drv = mc.data["drivers"].get(m, [])
                d = ", ".join(f"{n} ({r:+.2f})" for n, r in drv[:3])
                lines.append(f"| {m} | {q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g} | {d} |")
        fmc = next((c for c in self.checks if c.name == "failure_modes"), None)
        if fmc:
            lines += ["", "## Failure modes (user-defined)", "",
                      "| mode | criterion | nominal SF | P(fail) [95% CI] | SF P5 / P50 / min | margin driven by | origin |",
                      "|---|---|---|---|---|---|---|"]
            for m, v in fmc.data["modes"].items():
                crit = f"{v['metric']} {'>=' if v['fails_when'] == 'above' else '<='} {v['limit']}" if v["metric"] else "custom"
                lo, hi = v["p_fail_ci95"]
                drv = ", ".join(f"{n} ({r:+.2f})" for n, r in v["drivers"][:2])
                gov = " **(governing)**" if v.get("governing") else ""
                org = v["origin"] + (f"; ASSUMPTION: {v['assumption']}" if v["assumption"] else "")
                lines.append(f"| {m}{gov} | {crit} | {v['nominal_SF']:.3g} | {v['p_fail']*100:.3g}% [{lo*100:.2g}, {hi*100:.2g}] | "
                             f"{v['SF_p5']:.3g} / {v['SF_p50']:.3g} / {v['SF_min']:.3g} | {drv} | {org} |")
        tor = next((c for c in self.checks if c.name == "sensitivity"), None)
        if tor and tor.data.get("rows"):
            lines += ["", "## One-at-a-time sensitivity (trend report, not judged)", "",
                      "| parameter | metric | low | nominal | high |", "|---|---|---|---|---|"]
            for r in tor.data["rows"]:
                lines.append(f"| {r[0]} | {r[1]} | {r[2]:.4g} | {r[3]:.4g} | {r[4]:.4g} |")
        return "\n".join(lines) + "\n"

    def to_json(self):
        return json.dumps(dict(name=self.spec.name, ok=self.ok, summary=self.summary(),
                               checks=[c.__dict__ for c in self.checks]), default=float, indent=1)

    def __repr__(self):
        return self.summary()


# progress: the desktop app sets a hook to show which stage the battery is in (none by default)
_HOOK = None
STAGES = ["Baseline run", "Determinism", "Finite outputs", "Physical bounds", "Stability margin",
          "Time-step convergence", "Energy ledger", "Conservation", "Limit cases", "Knockouts", "Sensitivity",
          "Monte Carlo", "Failure modes", "Anchors"]


def set_progress_hook(fn):
    """fn(stage_index, stage_label, sub_done, sub_total) or None."""
    global _HOOK
    _HOOK = fn


def _stage(i, done=0, total=0):
    if _HOOK is not None:
        try:
            _HOOK(i, STAGES[i], done, total)
        except Exception:
            pass


def _call(spec, rep, params, dt):
    rep.runs += 1
    out = spec.run(dict(params), dt)
    return {k: float(v) for k, v in out.items()}


def _metric_names(spec, res):
    return spec.metrics or [k for k in res if not k.startswith("_")]


def _finite(res):
    return all(math.isfinite(v) for v in res.values())


@block("diagnostics", "verification", "any_sim", "full_battery", kind="builder",
       source="verification & validation practice (Roache 1998; Oberkampf & Roy 2010)",
       tags=("dt convergence", "knockout", "Monte Carlo", "ledger", "veto"))
def verify_all(spec, skip=()):
    """Run the full verification battery on a SimSpec. Returns a Report (report.ok, .markdown(), .to_json())."""
    rep = Report(spec)
    p0, dt0 = spec.params, spec.dt
    _stage(0)
    base = _call(spec, rep, p0, dt0)
    names = _metric_names(spec, base)

    # 1 determinism
    _stage(1)
    again = _call(spec, rep, p0, dt0)
    same = all(base[k] == again[k] or (math.isnan(base[k]) and math.isnan(again[k])) for k in base)
    rep.add("determinism", PASS if same else FAIL, "bit-identical repeat" if same else "repeat run differs")

    # 2 NaN/inf
    _stage(2)
    rep.add("finite_outputs", PASS if _finite(base) else FAIL,
            "all outputs finite" if _finite(base) else f"non-finite: {[k for k, v in base.items() if not math.isfinite(v)]}")

    # 3 bounds
    _stage(3)
    if spec.bounds:
        bad = [f"{m}={base[m]:.4g} not in [{lo}, {hi}]" for m, (lo, hi) in spec.bounds.items()
               if m in base and not (lo <= base[m] <= hi)]
        rep.add("physical_bounds", FAIL if bad else PASS, "; ".join(bad) or f"{len(spec.bounds)} bounds respected")
    else:
        rep.add("physical_bounds", SKIP, "no bounds declared")

    # 4 stability margin
    _stage(4)
    if spec.critical_dt:
        dc = spec.critical_dt(p0); mgn = dc / dt0
        st = FAIL if mgn < 1 else (WARN if mgn < spec.dt_accuracy_margin else PASS)
        rep.add("stability_margin", st, f"critical dt {dc:.3g} s / dt {dt0:.3g} s = {mgn:.1f}x", dict(margin=mgn))
    else:
        rep.add("stability_margin", SKIP, "no critical_dt provided")

    # 5 timestep convergence + 6 ledger at each level
    _stage(5)
    if "convergence" not in skip:
        levels = [base] + [_call(spec, rep, p0, dt0 / 2 ** i) for i in range(1, spec.conv_levels)]
        rows = []; worst = PASS
        for m in names:
            v = [lv[m] for lv in levels]
            e1, e2 = abs(v[0] - v[1]), abs(v[1] - v[2]) if len(v) > 2 else float("nan")
            scale = max(abs(v[-1]), 1e-12)
            if e1 <= 1e-12 * scale:
                rows.append((m, v[0], v[-1], float("inf"), 0.0)); continue
            p = math.log2(e1 / e2) if e2 > 0 else float("inf")
            if math.isfinite(p) and p > 0.2:
                vstar = v[-1] + (v[-1] - v[-2]) / (2 ** p - 1)
            else:
                vstar = v[-1]
            err = abs(v[0] - vstar) / max(abs(vstar), 1e-12)
            rows.append((m, v[0], vstar, p, err))
            if err > spec.conv_tol:
                worst = FAIL
            elif (not math.isfinite(p) or p < 0.5) and worst == PASS and e1 > 1e-4 * scale:
                worst = WARN          # converged in value but order not clean (switching/contact events). A change
                                      # under 0.01% between dt levels is converged: its order is round-off, not judged
        det = "; ".join(f"{m}: err {e*100:.2g}% (order {p:.2g})" for m, _, _, p, e in rows)
        rep.add("dt_convergence", worst, det, dict(rows=rows, dts=[dt0 / 2 ** i for i in range(spec.conv_levels)]))
        if "_ledger_rel" in base:
            worst_l = max(abs(lv["_ledger_rel"]) for lv in levels)
            rep.add("energy_ledger", PASS if worst_l < spec.ledger_tol else FAIL,
                    f"worst closure residual {worst_l*100:.2g}% across {len(levels)} dt levels (tol {spec.ledger_tol*100:.0f}%)")
        else:
            rep.add("energy_ledger", WARN, "sim reports no _ledger_rel: energy closure unverified")
    # 7 conservation
    _stage(7)
    for c in spec.conserved:
        if c in base:
            rep.add(f"conservation:{c}", PASS if abs(base[c]) < spec.conserved_tol else FAIL, f"{c} = {base[c]:.3g}")
        else:
            rep.add(f"conservation:{c}", FAIL, "declared conserved but not reported")

    # 8 limit cases
    _stage(8)
    for desc, ov, expect in spec.limits:
        r = _call(spec, rep, {**p0, **ov}, dt0); bad = []
        for m, e in expect.items():
            ok = e(r[m]) if callable(e) else abs(r[m] - e) <= 1e-6 * max(1.0, abs(e))
            if not ok:
                bad.append(f"{m}={r[m]:.4g}")
        rep.add(f"limit:{desc}", FAIL if bad else PASS, "; ".join(bad) or "matches expected limit")

    # 9 knockouts
    _stage(9)
    for kname, spec_k in spec.knockouts.items():
        ov, signs = (spec_k if isinstance(spec_k, tuple) else (spec_k, {}))
        r = _call(spec, rep, {**p0, **ov}, dt0)
        ch = {m: (r[m] - base[m]) / max(abs(base[m]), 1e-12) for m in names}
        dead = all(abs(x) < spec.knockout_dead_tol for x in ch.values())
        wrong = [m for m, s in signs.items() if s in "+-" and ((s == "+" and ch[m] <= 0) or (s == "-" and ch[m] >= 0))]
        # a knockout that removes a load path can legitimately send an output to infinity (e.g. a safety
        # factor with zero demand): that proves the block is alive, so it is a WARN note, not a failure
        st = FAIL if (dead or wrong) else (WARN if not _finite(r) else PASS)
        det = "DEAD: no metric changed (wiring bug?)" if dead else \
            ", ".join(f"{m} {x*100:+.3g}%" for m, x in ch.items())
        if not dead and not _finite(r):
            det += " | non-finite output with this term removed (load path gone)"
        if wrong:
            det += f" | WRONG SIGN on {wrong}"
        rep.add(f"knockout:{kname}", st, det, dict(change=ch))

    # 10 sensitivity (trend report)
    _stage(10)
    if spec.uncertain and "sensitivity" not in skip:
        rows = []
        for pn, rg in spec.uncertain.items():
            if callable(rg):
                continue
            lo, hi = rg
            rl = _call(spec, rep, {**p0, pn: lo}, dt0); rh = _call(spec, rep, {**p0, pn: hi}, dt0)
            for m in names:
                rows.append((pn, m, rl[m], base[m], rh[m]))
        rep.add("sensitivity", PASS, f"{len(spec.uncertain)} inputs swept low/high (trends reported, not judged)",
                dict(rows=rows))

    # 11 Monte Carlo
    _stage(11)
    if spec.uncertain and spec.mc_n and "monte_carlo" not in skip:
        rng = np.random.default_rng(spec.mc_seed)
        X, Y, nfail, outs = [], [], 0, []
        keys = list(spec.uncertain)
        for i_mc in range(spec.mc_n):
            if i_mc % max(1, spec.mc_n // 50) == 0:
                _stage(11, i_mc, spec.mc_n)
            s = {k: (v(rng) if callable(v) else rng.uniform(*v)) for k, v in spec.uncertain.items()}
            r = _call(spec, rep, {**p0, **s}, dt0)
            okb = _finite(r) and all(lo <= r[m] <= hi for m, (lo, hi) in spec.bounds.items() if m in r)
            if "_ledger_rel" in r and abs(r["_ledger_rel"]) >= spec.ledger_tol:
                okb = False
            if not okb:
                nfail += 1
            X.append([s[k] for k in keys]); Y.append([r[m] for m in names]); outs.append(r)
        X, Y = np.array(X), np.array(Y)
        pct = {m: list(np.nanpercentile(Y[:, j], [5, 50, 95])) for j, m in enumerate(names)}
        drivers = {}
        for j, m in enumerate(names):
            cors = []
            for i, k in enumerate(keys):
                cors.append((k, _spearman(X[:, i], Y[:, j])))
            drivers[m] = sorted(cors, key=lambda t: -abs(t[1]) if math.isfinite(t[1]) else 0)
        rate = nfail / spec.mc_n
        rep.add("monte_carlo", FAIL if rate > 0 else PASS,
                f"{spec.mc_n} samples, {nfail} failed finite/bounds/ledger ({rate*100:.1f}%)",
                dict(percentiles=pct, drivers=drivers, X=X.tolist(), Y=Y.tolist(), inputs=keys, metrics=names))
        rep.mc_outputs = outs; rep.mc_X = X; rep.mc_inputs = keys

    # 12 failure modes (design result: WARN if anything fails, never a verification FAIL)
    _stage(12)
    if spec.failure_modes:
        from .failure import assess_failure_modes
        fm = assess_failure_modes(spec.failure_modes, base, getattr(rep, "mc_outputs", ()),
                                  getattr(rep, "mc_X", None), getattr(rep, "mc_inputs", None))
        any_fail = any(v["nominal_failed"] or (v["n_failed"] > 0) for v in fm.values())
        unreach = [k for k, v in fm.items() if v.get("reach") and not v["reach"]["reachable"]]
        det = "; ".join(f"{k}: nominal SF {v['nominal_SF']:.3g}{' FAILED' if v['nominal_failed'] else ''}, "
                        f"P(fail) {v['p_fail']*100:.3g}%"
                        + (f" [UNREACHABLE: geometry max {v['reach']['max_possible']:.3g}]" if k in unreach else "")
                        for k, v in fm.items())
        rep.add("failure_modes", WARN if (any_fail or unreach) else PASS, det, dict(modes=fm, unreachable=unreach))
        rep.failure = fm

    # 13 anchors
    _stage(13)
    for m, (val, tol, src) in spec.anchors.items():
        err = abs(base[m] - val) / max(abs(val), 1e-12)
        rep.add(f"anchor:{m}", PASS if err <= tol else FAIL, f"{base[m]:.4g} vs {val:.4g} ({src}), err {err*100:.2g}%")
    return rep


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@block("diagnostics", "verification", "any_sim", "report_plot", kind="builder", source="matplotlib",
       tags=("report",))
def plot_report(rep, path):
    """Convergence, knockout and Monte Carlo figure for a Report (PNG/PDF by extension)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..viz import theme as T
    T.apply()
    C = T.SERIES
    conv = next((c for c in rep.checks if c.name == "dt_convergence"), None)
    kos = [c for c in rep.checks if c.name.startswith("knockout:")]
    mc = next((c for c in rep.checks if c.name == "monte_carlo"), None)
    f, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    if conv:
        for i, (m, v0, vs, p, e) in enumerate(conv.data["rows"]):
            ax[0].bar(i, max(e, 1e-16) * 100, color=C[i % 6])
        ax[0].set_yscale("log"); ax[0].axhline(rep.spec.conv_tol * 100, color="k", ls="--", lw=1)
        ax[0].set_xticks(range(len(conv.data["rows"]))); ax[0].set_xticklabels([r[0] for r in conv.data["rows"]], rotation=30, fontsize=7)
        ax[0].set_title("dt discretization error (%)")
    if kos:
        ms = list(kos[0].data["change"])
        w = 0.8 / max(len(ms), 1)
        for j, m in enumerate(ms):
            ax[1].bar(np.arange(len(kos)) + j * w, [k.data["change"][m] * 100 for k in kos], width=w, color=C[j % 6], label=m)
        ax[1].set_xticks(np.arange(len(kos)) + 0.4 - w / 2); ax[1].set_xticklabels([k.name[9:] for k in kos], rotation=30, fontsize=7)
        ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("knockout effect (%)"); ax[1].legend(fontsize=7)
    if mc and mc.data.get("Y"):
        Y = np.array(mc.data["Y"])
        ax[2].hist(Y[:, 0], bins=30, color=C[0]); ax[2].set_title(f"Monte Carlo: {mc.data['metrics'][0]}")
    top = T.header(f, rep.spec.name, "dt error, knockout effects, Monte Carlo spread",
                   status=("PASS VERIFIED" if rep.ok else "FAIL NOT VERIFIED"), tag="SIMBLOCKS // VERIFICATION")
    for a in ax:
        T.style_ticks(a)
    f.tight_layout(rect=(0, 0, 1, top))
    from ..viz.qa import check_figure, LayoutError
    probs = check_figure(f); T.QA_LOG[path] = probs
    if probs and T.QA_STRICT:
        plt.close(f)
        raise LayoutError(f"{path}: " + "; ".join(probs[:10]))
    f.savefig(path, facecolor=T.SURFACE); plt.close(f)
    return path

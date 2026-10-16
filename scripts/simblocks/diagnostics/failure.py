"""User-defined failure modes.

Failure is whatever the USER says it is for this part ("tip drops 5 mm permanently", "gets punctured",
"wheel loses contact for > 50 ms"), never an implicit "over yield". Each FailureMode turns one such
statement into a check on the sim's outputs, with a margin so near-misses are visible.

    FailureMode("tip drop", metric="tip_drop_perm_mm", limit=5.0, fails_when="above",
                origin="user: the ramp fails at a permanent 5 mm tip drop")

Margins: capacity/demand ratio SF = limit/value for 'above', value/limit for 'below' (SF < 1 = failed),
and a signed normalized margin (limit - value)/|limit|. A custom check can be any callable
out_dict -> (failed: bool, SF: float).

Assessment (assess_failure_modes) runs over the nominal case and every Monte Carlo sample:
P(fail) per mode with a Wilson 95% interval, governing mode (highest P(fail), ties by smallest SF seen),
SF percentiles, and the inputs that drive each mode's margin (rank correlation).
"""
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..registry import block


@dataclass
class FailureMode:
    name: str
    metric: str = None
    limit: float = None
    fails_when: str = "above"            # 'above' or 'below'
    check: Callable = None               # optional custom: out -> (failed, SF)
    origin: str = "user"                 # who defined it and how ('user: ...', 'assumption: ...')
    assumption: str = ""                 # any placeholder value the user has not confirmed
    reach_metric: str = None             # output giving the largest value the model geometry can EVER produce;
                                         # if it cannot reach the limit the mode is UNREACHABLE, not PASS

    def evaluate(self, out):
        if self.check is not None:
            failed, sf = self.check(out)
            return bool(failed), float(sf)
        v = out[self.metric]
        if self.fails_when == "above":
            sf = self.limit / v if v > 0 else math.inf
            return v >= self.limit, sf
        sf = v / self.limit if self.limit != 0 else math.inf
        return v <= self.limit, sf


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@block("diagnostics", "failure", "any_sim", "assessment", kind="builder",
       source="reliability counting with Wilson score interval", tags=("probability of failure", "margin", "governing mode"))
def assess_failure_modes(modes, nominal_out, mc_outputs=(), mc_inputs=None, input_names=None):
    """Evaluate user failure modes on the nominal run and all Monte Carlo samples. Returns dict per mode."""
    res = {}
    for fm in modes:
        f0, sf0 = fm.evaluate(nominal_out)
        fl, sfs = [], []
        for o in mc_outputs:
            f, s = fm.evaluate(o)
            fl.append(f); sfs.append(s)
        n = len(fl); k = int(sum(fl))
        sfs = np.array(sfs, float)
        drivers = []
        if n and mc_inputs is not None:
            X = np.asarray(mc_inputs, float)
            fin = np.isfinite(sfs)
            for i, nm in enumerate(input_names):
                drivers.append((nm, _rank_corr(X[fin, i], sfs[fin])))
            drivers.sort(key=lambda t: -abs(t[1]) if math.isfinite(t[1]) else 0)
        reach = None
        if fm.reach_metric and fm.reach_metric in nominal_out:
            rv = nominal_out[fm.reach_metric]
            reach = dict(max_possible=rv, reachable=(rv >= fm.limit) if fm.fails_when == "above" else (rv <= fm.limit))
        res[fm.name] = dict(reach=reach, nominal_failed=f0, nominal_SF=sf0, n=n, n_failed=k,
                            p_fail=k / n if n else float("nan"), p_fail_ci95=_wilson(k, n),
                            SF_p5=float(np.nanpercentile(sfs, 5)) if n else float("nan"),
                            SF_p50=float(np.nanpercentile(sfs, 50)) if n else float("nan"),
                            SF_min=float(np.nanmin(sfs)) if n else float("nan"),
                            drivers=drivers, origin=fm.origin, assumption=fm.assumption,
                            metric=fm.metric, limit=fm.limit, fails_when=fm.fails_when,
                            SF_samples=sfs.tolist())
    if res:
        # governing = highest P(fail); ties broken by the smallest margin seen (nominal or any sample)
        def worst_sf(m):
            v = [res[m]["nominal_SF"], res[m]["SF_min"]]
            v = [x for x in v if math.isfinite(x)]
            return min(v) if v else math.inf
        live = [m for m in res if not (res[m]["reach"] and not res[m]["reach"]["reachable"])] or list(res)
        gov = max(live, key=lambda m: (res[m]["p_fail"] if math.isfinite(res[m]["p_fail"]) else -1, -worst_sf(m)))
        for m in res:
            res[m]["governing"] = (m == gov)
    return res


def _rank_corr(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@block("diagnostics", "failure", "any_sim", "screening_prompts", source="failure-mode screening practice",
       tags=("elicitation", "candidate modes"))
def candidate_failure_modes(part_type):
    """Candidate failure statements to PROPOSE to the user (never apply silently). The user picks and sets limits."""
    lib = {
        "ramp": ["permanent tip/lip height drop above X mm", "puncture by the striking body",
                  "fold to the backstop", "mount bolt shear or tear-out", "lip dent depth above X mm"],
        "blade": ["tooth shear-off", "permanent bend above X deg", "hub bore cracking (brittle, Kt)",
                  "bolt-pattern tear-out", "burst above X rpm"],
        "shaft": ["permanent bend above X deg (runout)", "torsional yield at keyway", "bearing seat fretting"],
        "plate": ["permanent dish above X mm", "perforation", "first-ply failure (composite)", "delamination"],
        "drive": ["wheel loses contact for > X ms", "motor current above controller limit for > X s",
                  "battery sag below X V", "stall on a floor seam or step"],
        "mount": ["fastener shear", "thread strip", "bearing elongation above X mm", "net-section tension"],
    }
    return lib.get(part_type, sorted({s for v in lib.values() for s in v}))

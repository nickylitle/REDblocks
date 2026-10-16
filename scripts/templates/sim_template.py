"""TEMPLATE: copy this file, rename, fill the four marked sections. Keep the structure:
build() from blocks -> njit loop calling kernels -> metrics incl. _ledger_rel -> SimSpec -> verify_all.
Run:  python my_sim.py      (prints the verification report, writes out/<name>_verification.md/.png)
"""
import math, os, sys
import numpy as np
from numba import njit

SKILL_SCRIPTS = os.environ.get("SIMBLOCKS_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, SKILL_SCRIPTS)
from simblocks.materials import get, load_curve
from simblocks.diagnostics.verify import SimSpec, verify_all, plot_report
from simblocks.diagnostics.failure import FailureMode
from simblocks.integrators.explicit import critical_dt_spring
# (1) import the blocks you selected, e.g.
from simblocks.contact.penalty import penalty_linear
from simblocks.constitutive.plasticity import tabular_return, curve_to_spring_table

# ---------------------------------------------------------------- (2) parameters: every input named, SI units
DEFAULTS = dict(material="7075", m=1.3, v0=3.0, A=2e-5, L=0.03, t_end=2e-3)


def build(p):
    """All pre-loop physics from builder blocks. Nothing physical is computed inside the loop except kernels."""
    curve = load_curve(p["material"])                      # published fit, source travels with it
    k, acc_tab, F_tab = curve_to_spring_table(curve, p["A"], p["L"])
    return dict(k=k, acc_tab=acc_tab, F_tab=F_tab, curve_source=curve.source)


# ---------------------------------------------------------------- (3) the loop: kernels only, ledger always
@njit(cache=True)
def _loop(n, dt, m, v0, k, acc_tab, F_tab):
    x = 0.0; v = v0; xp = 0.0; acc = 0.0; W_pl = 0.0; Fmax = 0.0
    for i in range(n):
        xp_old = xp
        F, xp, acc = tabular_return(x, xp, acc, k, acc_tab, F_tab, 1)
        W_pl += F * (xp - xp_old)                             # every sink accumulated
        v -= F / m * dt                                       # semi-implicit Euler (velocity first)
        x += v * dt
        if F > Fmax:
            Fmax = F
    F, _, _ = tabular_return(x, xp, acc, k, acc_tab, F_tab, 1)
    E0 = 0.5 * m * v0 * v0
    E1 = 0.5 * m * v * v + (0.5 * F * F / k if F > 0 else 0.0)
    return xp, Fmax, -v, (E0 - E1 - W_pl) / max(E0, 1e-30)


def run(params=None, dt=1e-7):
    p = dict(DEFAULTS); p.update(params or {}); b = build(p)
    set_, Fmax, vreb, led = _loop(int(round(p["t_end"] / dt)), dt, p["m"], p["v0"], b["k"], b["acc_tab"], b["F_tab"])
    return dict(set_mm=set_ * 1e3, Fmax_N=Fmax, rebound=vreb, _ledger_rel=led)


def critical_dt(params):
    p = dict(DEFAULTS); p.update(params); b = build(p)
    return critical_dt_spring(b["k"], p["m"])


# ---------------------------------------------------------------- (4) verification spec: REQUIRED before reporting
SPEC = SimSpec(
    name="template impact", run=lambda p, dt: run(p, dt), params={}, dt=1e-7,
    uncertain={"v0": (2.0, 4.0), "A": (1.5e-5, 2.5e-5)},
    knockouts={"rigid": (dict(A=1.0), {"set_mm": "-"})},          # one per physics term
    limits=[("no speed -> no set", dict(v0=0.0), {"set_mm": 0.0})],
    bounds={"set_mm": (0.0, 30.0)}, critical_dt=critical_dt, mc_n=200,
    # (2b) the USER's failure definition, e.g. "fails at 1 mm permanent set" (ask; never assume yield = fail)
    failure_modes=[FailureMode("permanent set", metric="set_mm", limit=1.0, origin="user: <their words>")])

if __name__ == "__main__":
    rep = verify_all(SPEC)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, f"{SPEC.name.replace(' ', '_')}_verification.md"), "w").write(rep.markdown())
    plot_report(rep, os.path.join(out, f"{SPEC.name.replace(' ', '_')}_verification.png"))
    print(rep.markdown())
    print("material law:", build(DEFAULTS)["curve_source"])

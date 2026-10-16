"""Energy ledger (closure veto) and convergence checks."""
import numpy as np

from ..registry import block


@block("diagnostics", "energy", "general", "ledger", kind="builder", source="first law closure",
       tags=("veto", "validation"))
class EnergyLedger:
    """Accumulate every source and sink as power*dt, then check closure:
    sum(sources) = dKE + dPE_elastic + sum(sinks).  Usage inside a loop:
        L = EnergyLedger(); L.add('motor', P_in*dt, source=True); L.add('dent', P_dent*dt) ...
        L.close(E0, E1)  -> dict(residual, rel)"""

    def __init__(self):
        self.src = {}; self.snk = {}

    def add(self, name, dE, source=False):
        d = self.src if source else self.snk
        d[name] = d.get(name, 0.0) + dE

    def close(self, E_stored_start, E_stored_end, tol=0.05):
        ins = sum(self.src.values()); outs = sum(self.snk.values())
        res = ins - (E_stored_end - E_stored_start) - outs
        scale = max(abs(ins), abs(E_stored_end - E_stored_start) + abs(outs), 1e-12)
        return dict(sources=dict(self.src), sinks=dict(self.snk), dE_stored=E_stored_end - E_stored_start,
                    residual=res, rel=res / scale, ok=abs(res / scale) < tol)


@block("diagnostics", "convergence", "general", "dt_halving", kind="builder", source="Richardson / step halving",
       tags=("validation",))
def dt_convergence(run_metric, dt, factor=2.0, tol=0.01):
    """Run metric(dt) and metric(dt/factor). Returns dict(m1, m2, rel_change, ok)."""
    m1 = run_metric(dt); m2 = run_metric(dt / factor)
    rel = abs(m2 - m1) / max(abs(m2), 1e-30)
    return dict(m1=m1, m2=m2, rel_change=rel, ok=rel < tol)


@block("diagnostics", "convergence", "general", "richardson", source="Richardson extrapolation")
def richardson(m_h, m_h2, order=1, factor=2.0):
    """Extrapolated limit from results at step h and h/factor for a method of given order."""
    r = factor ** order
    return (r * m_h2 - m_h) / (r - 1)

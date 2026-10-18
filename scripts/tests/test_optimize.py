"""Optimizer: a constrained minimum with a closed-form answer."""
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.optimize import spec, optimize

M = """simblocks: 1
name: drop
dim: 1d
params: {k: 5000 N/m, m: 1 kg}
bodies:
  a: {dof: translation, mass: $m, v0: 2 m/s}
elements:
  pad: {type: kelvin_voigt, measure: a, one_sided: true, k: $k, c: 0}
outputs:
  stroke: {element: pad, quantity: measure, reduce: max, unit: mm}
  F: {element: pad, quantity: force, reduce: max}
failure:
  - {name: pad force, metric: F, limit: 600, fails_when: above, origin: "user: test"}
study:
  optimize:
    vary: {k: [1 kN/m, 1000 kN/m]}
    minimize: stroke
    subject_to: [{output: F, max: 500}]
    failure_modes: true
run: {t_end: 60 ms}
"""


def test_stiffest_pad_under_the_force_limit():
    s = load_system(M)
    G = spec(s.doc, s)
    r = optimize(s, G)
    assert r["found"]
    k = r["best"]["values"]["k"]
    assert k == pytest.approx((500 / 2) ** 2 / 1.0, rel=0.02)          # F = v sqrt(k m) = 500 N
    assert r["best"]["outputs"]["F"] <= 500 * 1.0005
    assert len(r["runs"]) < 150


def test_no_feasible_point_is_reported():
    s = load_system(M.replace("max: 500", "max: 1"))
    r = optimize(s, spec(s.doc, s))
    assert not r["found"] and "no trial met" in r["reason"]

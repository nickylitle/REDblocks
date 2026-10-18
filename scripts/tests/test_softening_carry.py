"""Thermal softening (Johnson-Cook) and a permanent set carried in from an earlier hit."""
import pytest
from simblocks.compiler import load_system

BASE = """simblocks: 1
name: t
dim: 1d
bodies:
  a: {{dof: translation, mass: 1 kg}}
elements:
  push: {{type: force, measure: a, pulse: {{kind: ramp, to: 900 N, duration: 50 ms}}}}
  bar: {{type: elastoplastic, measure: a, table: {{k: 1e6, acc_tab: [0, 1], F_tab: [500, 500]}}{extra}}}
  damp: {{type: dashpot, measure: a, c: 2000}}
outputs:
  F: {{element: bar, quantity: force, reduce: max}}
  set: {{element: bar, quantity: plastic_set, reduce: final}}
run: {{t_end: 60 ms}}
"""


def test_softening_scales_the_yield():
    cold = load_system(BASE.format(extra="")).run()
    hot = load_system(BASE.format(extra=", softening: {T: 400 C, material: 4140_normalized}")).run()
    fac = 1 - ((673.15 - 293.15) / (1793 - 293.15)) ** 1.0
    assert cold["F"] == pytest.approx(500, rel=1e-3)
    assert hot["F"] == pytest.approx(500 * fac, rel=1e-3)


def test_set_carried_in():
    """Below yield the carried set stays put: the bar springs back to it, not to zero."""
    m = BASE.format(extra=", set0: 1 mm").replace("to: 900 N", "to: 400 N").replace("mass: 1 kg}", "mass: 1 kg, x0: 1 mm}").replace(
        "  set: {element: bar, quantity: plastic_set, reduce: final}",
        "  set: {element: bar, quantity: plastic_set, reduce: final}\n  x: {body: a, quantity: position, reduce: final}")
    o = load_system(m).run()
    assert o["set"] == pytest.approx(1e-3, rel=1e-9)
    assert o["x"] == pytest.approx(1e-3 + 400 / 1e6, rel=1e-3)

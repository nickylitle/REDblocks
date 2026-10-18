"""Modes and static solve through the solver itself: checked against closed forms."""
import math

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.linear import modes, static


def S(t):
    return load_system("simblocks: 1\nname: t\n" + t)


def test_two_mass_chain_frequencies():
    s = S("""dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg}
  b: {dof: translation, mass: 2 kg}
elements:
  k1: {type: spring, measure: a, k: 1000}
  k2: {type: spring, measure: b - a, k: 500}
outputs:
  x: {body: a, quantity: position}
run: {t_end: 1 s}
""")
    r = modes(s.nominal)
    K = np.array([[1500, -500], [-500, 500.0]]); Mm = np.diag([1, 2.0])
    w = np.sqrt(np.sort(np.linalg.eigvals(np.linalg.solve(Mm, K)).real))
    assert [m["omega"] for m in r["modes"]] == pytest.approx(list(w), rel=1e-5)
    assert r["rigid"] == 0


def test_free_body_is_a_rigid_mode_and_gear_train_frequency():
    s = S("""dim: 1d
bodies:
  p: {dof: rotation, inertia: 2e-5 kg m^2}
  w: {dof: rotation, inertia: 1.8e-4 kg m^2}
elements:
  mesh: {type: gear, measure: p + w, module: 1 mm, teeth: [12, 36], face_width: 8 mm}
outputs:
  x: {body: p, quantity: velocity}
run: {t_end: 10 ms}
""")
    r = modes(s.nominal)
    k = s.nominal.arrays["P"][0, 0]
    inv = 0.006 ** 2 / 2e-5 + 0.018 ** 2 / 1.8e-4
    assert r["rigid"] == 1                                    # the pair turns together freely
    assert r["modes"][0]["omega"] == pytest.approx(math.sqrt(k * inv), rel=1e-5)


def test_pendulum_2d_with_a_pin():
    """A rod on a pin under gravity: omega^2 = m g d / (I_cm + m d^2)."""
    s = S("""dim: 2d
gravity: 9.81 m/s^2
bodies:
  g: {ground: true, points: {o: [0, 0]}}
  rod: {dof: planar, mass: 1 kg, inertia: 0.01 kg m^2, at: [0, -0.2 m], points: {top: [0, 0.2 m]}}
elements:
  hinge: {type: pin, measure: [g.o, rod.top], k: 1e7, c: 0}
outputs:
  y: {body: rod, quantity: y}
run: {t_end: 1 s}
""")
    r = modes(s.nominal)
    w_exp = math.sqrt(1 * 9.81 * 0.2 / (0.01 + 1 * 0.2 ** 2))
    assert r["modes"][0]["omega"] == pytest.approx(w_exp, rel=1e-3)


def test_static_spring_under_load_and_contact():
    s = S("""dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg}
elements:
  push: {type: force, measure: a, P: -50 N}
  floor: {type: hertz, measure: -a, K: 1e9, n: 1.5}
  k: {type: spring, measure: a, k: 1e4}
outputs:
  F: {element: floor, quantity: force, reduce: max}
run: {t_end: 1 s}
""")
    r = static(s.nominal)
    assert r["converged"]
    x = r["moved"]["a"]
    assert 1e4 * x + (-1e9 * (-x) ** 1.5 if x < 0 else 0) == pytest.approx(-50, rel=1e-6)
    assert r["outputs"]["F"] == pytest.approx(1e9 * (-x) ** 1.5, rel=1e-6)


def test_static_reports_a_free_body():
    s = S("""dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg}
elements:
  push: {type: force, measure: a, P: 5 N}
outputs:
  x: {body: a, quantity: position}
run: {t_end: 1 s}
""")
    r = static(s.nominal)
    assert not r["converged"] and r["free"] == ["a"]


def test_static_bolt_shares_load():
    s = S("""dim: 1d
bodies:
  lid: {dof: translation, mass: 0.05 kg}
elements:
  pull: {type: force, measure: lid, P: 2 kN}
  hold: {type: bolt, measure: lid, size: M5, grip: 6 mm, grade: "12.9", preload: 4 kN}
outputs:
  Fb: {element: hold, quantity: bolt_force, reduce: final}
run: {t_end: 1 ms}
""")
    r = static(s.nominal)
    kb, km = s.nominal.arrays["P"][1, 0], s.nominal.arrays["P"][1, 1]
    assert r["outputs"]["Fb"] == pytest.approx(4000 + kb / (kb + km) * 2000, rel=1e-6)


def test_spin_body_3d_torsion():
    s = S("""dim: 3d
bodies:
  g: {ground: true}
  disc: {dof: spin, axis: z, inertia: 0.002 kg m^2}
elements:
  shaft: {type: spring, measure: disc.spin, k: 50}
outputs:
  w: {body: disc, quantity: omega}
run: {t_end: 1 s}
""")
    r = modes(s.nominal)
    assert r["modes"][0]["omega"] == pytest.approx(math.sqrt(50 / 0.002), rel=1e-5)

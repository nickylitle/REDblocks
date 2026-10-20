"""The nonlinear (piecewise-linear) resistor and a driven current load (2026-10-20)."""
import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

RC = """simblocks: 1
name: t
dim: 1d
circuit:
  parts:
    c: {type: capacitor, between: [a, gnd], C: 1 uF, v0: 2 V}
    x: %s
outputs:
  v: {part: c, quantity: voltage, reduce: final}
  i: {part: x, quantity: current, reduce: final}
run: {t_end: 2 ms}
"""


def test_two_points_is_a_resistor():
    a = load_system(RC % "{type: resistor, between: [a, gnd], R: 1 kohm}").run()
    b = load_system(RC % "{type: nonlinear_resistor, between: [a, gnd], iv: [[-5, -5 mA], [5, 5 mA]]}").run()
    assert b["v"] == pytest.approx(a["v"], rel=1e-12) and b["v"] == pytest.approx(2 * math.exp(-2), rel=1e-4)


def test_segments_and_extrapolation():
    # 1 kohm below 1 V, 10 kohm above (carried on past the last point): the discharge follows both pieces
    o = load_system(RC % "{type: nonlinear_resistor, between: [a, gnd], iv: [[0, 0], [1, 1 mA], [1.5, 1.05 mA]]}").run()
    # analytic: from 2 V on the 10 kohm slope (i = 0.9 mA + v / 10k) down to 1 V, then 1 kohm
    C, g2, i2 = 1e-6, 1e-4, 0.9e-3
    t1 = C / g2 * math.log((2 + i2 / g2) / (1 + i2 / g2))           # time to fall to 1 V
    v_end = math.exp(-(2e-3 - t1) / (1e3 * C)) if t1 < 2e-3 else None
    assert o["v"] == pytest.approx(v_end, rel=2e-3)


def test_bad_tables_are_refused():
    for iv in ("[[1, 0]]", "[[1, 0], [0, 1 mA]]", "[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]"):
        with pytest.raises(SpecError):
            load_system(RC % ("{type: nonlinear_resistor, between: [a, gnd], iv: %s}" % iv)).run()


def test_chua_start_matches_scipy():
    import os
    y = open(os.path.join(os.path.dirname(__file__), "..", "examples", "learn", "1d_6_chua_circuit.yaml")).read()
    y = y.replace("run: {t_end: 60 ms}", "run: {t_end: 2 ms}")
    o, H = load_system(y).run(history=True, n_log=4000)
    C1, C2, L, R0, R, Ga, Gb, E = 10e-9, 100e-9, 18e-3, 12.5, 1800.0, -0.757e-3, -0.409e-3, 1.0
    def f(t, x):
        v1, v2, i = x
        g = Gb * v1 + 0.5 * (Ga - Gb) * (abs(v1 + E) - abs(v1 - E))
        return [((v2 - v1) / R - g) / C1, ((v1 - v2) / R + i) / C2, (-v2 - R0 * i) / L]
    t = np.asarray(H["t"])
    ref = solve_ivp(f, (0, t[-1]), [0.1, 0, 0], method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True).sol(t)
    assert np.abs(np.asarray(H["c1.voltage"]) - ref[0]).max() < 2e-3          # volts, on a 4 V swing


def test_a_current_load_can_be_driven():
    y = """simblocks: 1
name: t
dim: 1d
control:
  draw: "= 2 if t > 0.5e-3 else 0"
circuit:
  parts:
    s: {type: voltage_source, between: [a, gnd], V: 10 V, R: 1 ohm}
    ld: {type: current_load, between: [a, gnd], I: {control: draw, init: 0}}
outputs:
  i: {part: ld, quantity: current, reduce: final}
  v: {part: s, quantity: voltage, reduce: final}
run: {t_end: 1 ms}
"""
    o = load_system(y).run()
    assert o["i"] == pytest.approx(2.0, rel=1e-9) and o["v"] == pytest.approx(8.0, rel=1e-6)

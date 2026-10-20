"""Transfer-function and sample-and-hold control blocks (2026-10-20)."""
import math
import numpy as np
import pytest
from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

BASE = """simblocks: 1
name: t
dim: 1d
bodies: {m: {dof: translation, mass: 1 kg}}
elements: {s: {type: spring, k: 1 N/m, measure: m}}
control:
%s
outputs: {x: {body: m, quantity: position}}
run: {t_end: 1 s, dt: 0.1 ms}
"""


def hist(ctl):
    o, H = load_system(BASE % ctl).run(history=True, n_log=10001)
    return np.asarray(H["t"]), H


def test_first_order_matches_its_step_response():
    t, H = hist('  u: "= 1 if t > 0.1 else 0"\n  y: {transfer: u, num: [1], den: [0.05, 1]}')
    y = np.asarray(H["control.y"]); ref = np.where(t > 0.1, 1 - np.exp(-(t - 0.1) / 0.05), 0.0)
    assert np.abs(y - ref).max() < 3e-3                 # the one-step input timing, not the method


def test_second_order_overshoot_and_frequency():
    wn, z = 40.0, 0.2
    t, H = hist(f'  u: "= 1 if t > 0.1 else 0"\n  y: {{transfer: u, num: [{wn**2}], den: [1, {2*z*wn}, {wn**2}]}}')
    y = np.asarray(H["control.y"])
    assert y.max() - 1 == pytest.approx(math.exp(-z * math.pi / math.sqrt(1 - z * z)), rel=0.01)   # 52.7 % overshoot
    tp = t[np.argmax(y)] - 0.1
    assert tp == pytest.approx(math.pi / (wn * math.sqrt(1 - z * z)), rel=0.01)
    assert y[-1] == pytest.approx(1.0, abs=2e-3)


def test_lead_with_direct_feedthrough():
    # (s + 10) / (s + 100): jumps by 1 at the step, settles to 0.1
    t, H = hist('  u: "= 1 if t > 0.1 else 0"\n  y: {transfer: u, num: [1, 10], den: [1, 100]}')
    y = np.asarray(H["control.y"])
    assert y[np.searchsorted(t, 0.1002)] == pytest.approx(1.0, abs=0.03) and y[-1] == pytest.approx(0.1, abs=1e-3)


def test_hold_samples_at_its_period():
    t, H = hist('  ramp: "= t"\n  held: {hold: ramp, period: 10 ms}')
    h = np.asarray(H["control.held"])
    k = np.searchsorted(t, [0.0349, 0.0351, 0.5049])
    assert h[k[0]] == pytest.approx(0.03, abs=2e-4) and h[k[1]] == pytest.approx(0.03, abs=2e-4)
    assert h[k[2]] == pytest.approx(0.50, abs=2e-4)
    assert len(np.unique(np.round(h, 9))) == pytest.approx(101, abs=2)


def test_bad_transfer_is_refused():
    for bad in ('{transfer: u, num: [1, 2, 3], den: [1, 1]}', '{transfer: u, num: [1], den: [1, 1, 1, 1, 1]}',
                '{transfer: u, num: [1], den: [2]}', '{hold: u, period: 0}'):
        with pytest.raises(SpecError):
            load_system(BASE % ('  u: "= t"\n  y: ' + bad)).run()

"""Mass on 1D elements: lumped automatically, or resolved along the element with segments."""
import math

import pytest

from simblocks.compiler.system import load_system


def model(extra, tip=1.0, k=1000.0, ms=1.0, t_end=0.4):
    return load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  tip: {{dof: translation, mass: {tip} kg, x0: 0.01 m}}
elements:
  s: {{type: spring, k: {k} N/m, mass: {ms} kg, measure: tip{extra}}}
outputs:
  tq: {{body: tip, quantity: position, reduce: first_time_below, value: 0}}
  F: {{element: s, quantity: force, reduce: max_abs}}
  st: {{element: s, quantity: measure, reduce: max}}
run: {{t_end: {t_end} s, dt: 2e-6 s}}
""")


def _freq(text, t_end=4.0):
    import numpy as np
    o, H = load_system(text.replace("t_end: 0.4 s", f"t_end: {t_end} s")).run(history=True, n_log=40000)
    t, x = H["t"], H["tip.position"] - H["tip.position"].mean()
    N = len(x) * 16
    f = np.fft.rfftfreq(N, t[1] - t[0]); X = np.abs(np.fft.rfft(x * np.hanning(len(x)), N))
    return 2 * np.pi * f[np.argmax(X)]


def test_spring_mass_rayleigh_lumping_and_segments_converge_to_exact():
    """Fixed-free spring of mass m_s with tip mass M: exact fundamental beta tan(beta) = m_s / M,
    w = beta sqrt(k / m_s). One element lumps m_s/3 at the tip (Rayleigh, 0.7% high here); segments resolve it."""
    k, ms, M = 1000.0, 1.0, 1.0
    w_exact = 0.8603335890 * math.sqrt(k / ms)                   # beta tan(beta) = 1
    w1 = _freq(_seg_text(1).replace(", segments: 1", ""))
    assert w1 == pytest.approx(math.sqrt(k / (M + ms / 3)), rel=2e-3)
    assert _freq(_seg_text(20)) == pytest.approx(w_exact, rel=1e-3)


def _seg_text(n):
    return f"""simblocks: 1
name: t
dim: 1d
bodies:
  tip: {{dof: translation, mass: 1 kg, x0: 0.01 m}}
elements:
  s: {{type: spring, k: 1000 N/m, mass: 1 kg, segments: {n}, measure: tip}}
outputs:
  tq: {{body: tip, quantity: position, reduce: first_time_below, value: 0}}
  F: {{element: s, quantity: force, reduce: max_abs}}
  st: {{element: s, quantity: measure, reduce: max}}
  W: {{element: s, quantity: absorbed_work}}
run: {{t_end: 0.4 s, dt: 2e-6 s}}
"""


def test_segmented_spring_outputs_and_ledger():
    o = load_system(_seg_text(8)).run()
    assert o["st"] == pytest.approx(0.01, rel=1e-6)            # total stretch from the probe, starting value
    assert o["_ledger_rel"] < 0.01


def test_free_free_spring_mass_keeps_momentum():
    """Two bodies joined by a massive spring: the spring's mass is split half and half, total momentum kept."""
    S = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg, v0: 2 m/s}
  b: {dof: translation, mass: 1 kg}
elements:
  s: {type: kelvin_voigt, k: 1e4 N/m, c: 5 N s/m, mass: 0.5 kg, measure: a - b}
outputs:
  va: {body: a, quantity: velocity}
  vb: {body: b, quantity: velocity}
run: {t_end: 0.5 s}
""")
    o = S.run()
    assert (1.25 * o["va"] + 1.25 * o["vb"]) == pytest.approx(1.25 * 2, rel=1e-9)
    assert S.nominal.arrays["m"].tolist() == [1.25, 1.25]


def test_lever_mass_goes_to_inertia():
    S = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  rotor: {dof: rotation, inertia: 0.01 kg m^2}
elements:
  arm: {type: spring, k: 100, mass: 0.3 kg, measure: 0.1 * rotor}
outputs:
  w: {body: rotor, quantity: velocity}
run: {t_end: 0.1 s}
""")
    assert S.nominal.arrays["m"][0] == pytest.approx(0.01 + 0.3 / 3 * 0.01)

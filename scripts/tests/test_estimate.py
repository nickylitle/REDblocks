"""Parameter estimation from test data (2026-10-20): known parameters come back from noisy synthetic data."""
import numpy as np
import pytest
from simblocks.compiler import load_system
from simblocks.diagnostics.estimate import estimate

Y = """simblocks: 1
name: msd
dim: 1d
params:
  k: {value: 400 N/m}
  c: {value: 0.4 N s/m}
bodies:
  mass: {dof: translation, mass: 1 kg, x0: 10 mm}
elements:
  spring: {type: spring, k: $k, measure: mass}
  damper: {type: dashpot, c: $c, measure: mass}
outputs:
  x: {body: mass, quantity: position, reduce: max}
run: {t_end: 3 s}
"""


def test_recovers_stiffness_and_damping_from_noisy_data():
    s = load_system(Y)
    o, H = s.run({"k": 520.0, "c": 1.3}, history=True, n_log=3001)
    t = np.asarray(H["t"]); x = np.asarray(H["mass.position"])
    xm = x + np.random.default_rng(1).normal(0, 2e-4, len(t))          # 2 % of the swing as noise
    r = estimate(s, {"t": t[::10], "y": xm[::10]}, "mass.position", {"k": ["100 N/m", "2000 N/m"], "c": [0.01, 5]})
    assert r["values"]["k"] == pytest.approx(520, rel=2e-3) and r["values"]["c"] == pytest.approx(1.3, rel=0.03)
    assert abs(r["values"]["k"] - 520) < 4 * r["std"]["k"] + 0.5           # the uncertainty is honest
    assert r["rms"] == pytest.approx(2e-4, rel=0.2) and r["r2"] > 0.99 and not r["at_bound"]


def test_csv_with_units(tmp_path):
    s = load_system(Y)
    o, H = s.run({"k": 300.0}, history=True, n_log=301)
    p = tmp_path / "test.csv"
    np.savetxt(p, np.c_[np.asarray(H["t"]) * 1e3, np.asarray(H["mass.velocity"]) * 1e3], delimiter=",",
               header="time_ms,v_mm_s", comments="")
    r = estimate(s, str(p), "mass.velocity", {"k": [100, 1000]}, time_col="time_ms", y_col="v_mm_s",
                 time_unit="ms", unit="mm/s")
    assert r["values"]["k"] == pytest.approx(300, rel=1e-6)


def test_data_past_the_run_end_is_an_error():
    s = load_system(Y)
    with pytest.raises(ValueError):
        estimate(s, {"t": [0, 1, 5], "y": [0, 0, 0]}, "mass.position", {"k": [100, 1000]})

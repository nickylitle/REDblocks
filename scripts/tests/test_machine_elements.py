import math
import pytest
from simblocks.structures.machine_elements import (shaft_critical_speed, bearing_life, press_fit, key_capacity,
    helical_spring, belt_drive, crack_check, notch_sensitivity, lap_joint_shear, set_screw_holding, chain_drive)


def test_critical_speed_single_disc():
    E, d, L, m = 200e9, 0.012, 0.2, 2.0
    I = math.pi * d ** 4 / 64
    r = shaft_critical_speed(E, d, L, rho=1e-9, masses=[m], positions=[L / 2])
    w = math.sqrt(48 * E * I / (m * L ** 3))
    assert r["rpm_rayleigh"] == pytest.approx(w * 60 / (2 * math.pi), rel=1e-6)
    assert r["rpm_dunkerley"] == pytest.approx(r["rpm_rayleigh"], rel=1e-6)    # one mass: both exact


def test_press_fit_same_material_solid_shaft():
    E, d, D, dl = 200e9, 0.02, 0.04, 20e-6
    r = press_fit(dl, d, D, 0.02, E, E, mu=0.15)
    p = (dl / 2) * E * (D ** 2 - d ** 2) / (d * D ** 2)
    assert r["p"] == pytest.approx(p, rel=1e-9)
    assert r["slip_torque"] == pytest.approx(math.pi * 0.15 * p * d * d * 0.02 / 2, rel=1e-9)


def test_spring_and_bearing_and_key():
    s = helical_spring(0.002, 0.016, 8, G=79.3e9)
    assert s["k"] == pytest.approx(79.3e9 * 0.002 ** 4 / (8 * 0.016 ** 3 * 8))
    assert s["K_wahl"] == pytest.approx((4 * 8 - 1) / (4 * 8 - 4) + 0.615 / 8)
    b = bearing_life(10e3, 1e3, 1000)
    assert b["L_rev"] == pytest.approx(1e9) and b["L_hours"] == pytest.approx(1e9 / 60000)
    k = key_capacity(0.02, 0.006, 0.006, 0.02, 500e6)
    assert k["T"] == min(k["T_shear"], k["T_crush"])


def test_belt_chain_crack_notch_bond():
    b = belt_drive(100.0, 0.05, math.pi, mu=0.3)
    e = math.exp(0.3 * math.pi)
    assert b["F1"] / b["F2"] == pytest.approx(e) and b["F1"] + b["F2"] == pytest.approx(200.0)
    c = chain_drive(10.0, 0.00635, 15)
    assert c["D"] == pytest.approx(0.00635 / math.sin(math.pi / 15))
    k = crack_check(50e6, 300e6, 1e-3, "center")
    assert k["K"] == pytest.approx(300e6 * math.sqrt(math.pi * 1e-3))
    assert k["a_critical"] == pytest.approx((50e6 / 300e6) ** 2 / math.pi)
    n = notch_sensitivity(2.0, 0.003, 690e6)
    assert 0.7 < n["q"] < 0.95 and n["Kf"] == pytest.approx(1 + n["q"])
    j = lap_joint_shear(1000, 0.02, 0.02, 0.002, 70e9, 0.002, 70e9, 1e9, 2e-4)
    x = math.sqrt(1e9 / 2e-4 * 2 / (70e9 * 0.002)) * 0.02
    assert j["peak_factor"] == pytest.approx(x / 2 / math.tanh(x / 2), rel=1e-9)     # balanced Volkersen
    s = set_screw_holding(0.25 * 0.0254, 0.01)
    assert s["F"] == pytest.approx(1000 * 4.448222)


def test_elastoplastic_series_adds_compliance_at_equal_force():
    import numpy as np
    from simblocks.constitutive.plasticity import elastoplastic_series
    a = dict(k=1e6, acc_tab=np.array([0, 1e-3]), F_tab=np.array([100.0, 300.0]))
    b = dict(k=4e6, acc_tab=np.array([0, 1e-3]), F_tab=np.array([200.0, 400.0]))
    r = elastoplastic_series(a, b)
    assert r["k"] == pytest.approx(0.8e6)
    F = 250.0
    xa = (F - 100) / 200 * 1e-3; xb = (F - 200) / 200 * 1e-3
    assert np.interp(F, r["F_tab"], r["acc_tab"]) == pytest.approx(xa + xb, rel=1e-6)
    assert r["F_cap"] == 300.0


def test_flapjack_wedge_model_runs_and_balances():
    from simblocks.compiler import load_system
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "examples", "systems3d", "flapjack_wedge_strike.yaml")
    o = load_system(p).run()
    assert abs(o["_ledger_rel"]) < 1e-3
    assert 0 < o["gouge_mm"] < 5 and o["E_blade"] > 0 and o["tip_drop_mm"] < 5

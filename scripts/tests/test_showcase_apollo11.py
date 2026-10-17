"""The Apollo 11 showcase: the Saturn V ascent and the lunar module's powered descent, checked against NASA."""
import os

import pytest

from simblocks.compiler.system import load_system

HERE = os.path.join(os.path.dirname(__file__), "..", "examples", "showcase", "apollo11")


def run(name, overrides=None):
    return load_system(os.path.join(HERE, name)).run(overrides)


def test_ascent_matches_the_flight():
    """Saturn V AS-506 against NASA's postflight trajectory (Boeing D5-15560-6) and flight evaluation report."""
    o = run("apollo11_1_ascent.yaml")
    assert o["h_sep1"] == pytest.approx(66.7, rel=0.02)          # S-IC/S-II separation, km
    assert o["x_sep1"] == pytest.approx(95.1, rel=0.02)
    assert o["v_sep1"] == pytest.approx(2773.9, rel=0.01)
    assert o["fpa_sep1"] == pytest.approx(19.02, abs=0.3)
    assert o["v_sep2"] == pytest.approx(6916.6, rel=0.005)       # S-II/S-IVB separation, in-plane
    assert o["x_sep2"] == pytest.approx(1623.4, rel=0.01)
    assert o["T_eco"] == pytest.approx(699.33, abs=6.0)          # S-IVB guidance cutoff
    assert o["m_orbit"] == pytest.approx(135102, rel=0.01)
    assert o["vE_orbit"] == pytest.approx(7389.5, rel=0.002)
    assert o["_ledger_rel"] < 1e-3


def test_landing_follows_nasas_descent_plan():
    """The guided braking phase against NASA's premission trajectory (Bennett, TM X-58040 table I)."""
    o = run("apollo11_2_landing.yaml")
    for t, v in ((26, 5529), (176, 4000), (258, 3065), (384, 1456), (402, 1315), (506, 506)):
        assert o[f"v_{t:03d}"] == pytest.approx(v, abs=max(0.04 * v, 60)), t     # inertial speed, ft/s
    for t, h in ((26, 48725), (176, 44934), (258, 39201), (384, 24639)):
        assert o[f"h_{t:03d}"] == pytest.approx(h, rel=0.08), t                   # altitude, ft
    assert o["t_throttle"] == pytest.approx(386, abs=15)          # throttle recovery (flight: 386 s, plan 384 s)
    assert o["t_hg"] == pytest.approx(506.8, abs=10)              # high gate, P64 (flight 506.8 s, plan 506 s)
    assert o["t_touchdown"] == pytest.approx(754.7, abs=35)       # touchdown 102:45:39.9 GET
    assert o["_ledger_rel"] < 1e-3


def test_crew_landing_runs_longer_than_the_automatic_one():
    """NASA: the flown landing took 52-62 s longer than the automatic one would have (TM X-58040, fig. 26)."""
    flown = run("apollo11_2_landing.yaml")
    auto = run("apollo11_2_landing.yaml", {"crew": 0})
    assert flown["t_touchdown"] - auto["t_touchdown"] == pytest.approx(57, abs=15)
    assert flown["prop_left"] < auto["prop_left"]


def test_storable_propellant_chemistry_matches_cantera():
    """N2O4 / Aerozine 50 at O/F 1.6 and 110 psia: the chamber temperature Cantera gives for the same species."""
    from simblocks.actuators.rocket import ideal_performance
    r = ideal_performance("A50", 1.6, 110 * 6894.757, 47.5, "N2O4")
    assert r["Tc"] == pytest.approx(3059.98, abs=0.5)
    assert 320 < r["Isp_vac"] < 345

"""Levers (study.sensitivity): each input nudged up and down, outputs' derivatives and elasticities ranked."""
import time

import pytest
import yaml

from simblocks.app.api import Api

MODEL = open(__file__.replace("tests/test_sensitivity.py", "examples/learn/1d_1_mass_spring_damper.yaml")).read()


def run_levers(vary, of, step="1 %"):
    d = yaml.safe_load(MODEL); d.pop("editor", None)
    d["study"] = {"sensitivity": {"vary": vary, "of": of, "step": step}}
    A = Api(); t = yaml.safe_dump(d, sort_keys=False)
    assert A.study_start(t, 1)["ok"]
    while True:
        p = A.study_progress()
        if not p["running"]:
            return p
        time.sleep(0.05)


def test_levers_rank_by_elasticity_and_match_the_physics():
    p = run_levers(["k", "c", "m", "x_start"], ["peak_spring_N", "first_trough_mm"])
    t = p["sensitivity"]["table"]
    pk = {r["lever"]: r for r in t["peak_spring_N"]["rows"]}
    assert pk["k"]["elasticity"] == pytest.approx(1.0, abs=1e-6)          # peak spring force = k x0
    assert pk["x_start"]["elasticity"] == pytest.approx(1.0, abs=1e-6)
    assert pk["c"]["elasticity"] == pytest.approx(0.0, abs=1e-9)
    assert pk["k"]["dydx"] == pytest.approx(0.01, rel=1e-6)                # N per N/m: x0 = 10 mm
    rows = t["first_trough_mm"]["rows"]
    assert rows[0]["lever"] == "x_start"                                   # the trough scales with the start
    assert [abs(r["elasticity"]) for r in rows] == sorted((abs(r["elasticity"]) for r in rows), reverse=True)
    assert len(p["results"]) == 1 and p["total"] == 9                      # nominal + 2 per lever


def test_levers_reject_a_bad_name():
    d = yaml.safe_load(MODEL); d.pop("editor", None)
    d["study"] = {"sensitivity": {"vary": ["nope"]}}
    r = Api().study_start(yaml.safe_dump(d, sort_keys=False), 1)
    assert not r["ok"] and "nope" in r["error"]


def _study(d):
    A = Api(); r = A.study_start(yaml.safe_dump(d, sort_keys=False), 1)
    assert r["ok"], r
    while True:
        p = A.study_progress()
        if not p["running"]:
            return p
        time.sleep(0.05)


def test_levers_give_a_first_order_spread_from_uncertainty_ranges():
    d = yaml.safe_load(MODEL); d.pop("editor", None)
    d["uncertain"] = {"k": ["360 N/m", "440 N/m"]}
    d["study"] = {"sensitivity": {"vary": ["k", "c"], "of": ["peak_spring_N"]}}
    sp = _study(d)["sensitivity"]["table"]["peak_spring_N"]["spread"]
    assert sp["sd"] == pytest.approx(0.01 * 80 / 12 ** 0.5, rel=1e-6)          # dF/dk = x0, sigma_k = range / sqrt 12
    assert sp["share"]["k"] == pytest.approx(1.0)


def test_goal_seek_finds_the_input_for_a_target():
    d = yaml.safe_load(MODEL); d.pop("editor", None)
    d["study"] = {"goal_seek": {"vary": "c", "output": "first_trough_mm", "target": -5, "tolerance": "0.1 %"}}
    g = _study(d)["goal"]
    assert g["found"] and abs(g["y"] + 5) <= 0.005 and len(g["runs"]) <= 10

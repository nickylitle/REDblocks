"""Weight budget, scenarios and model compare."""
import pytest
from simblocks.compiler import load_system
from simblocks.app.api import Api

M = """simblocks: 1
name: bot
dim: 1d
params: {v: 2 m/s}
bodies:
  chassis: {dof: translation, mass: 800 g, v0: $v}
  wedge: {dof: translation, mass: 150 g}
elements:
  mount: {type: kelvin_voigt, measure: wedge - chassis, k: 1e5, c: 20}
  wall: {type: kelvin_voigt, measure: wedge, one_sided: true, k: 1e6, c: 50}
outputs:
  F: {element: wall, quantity: force, reduce: max}
budget:
  limit: 3 lb
  items: {weapon: 450 g, wiring: 25 g}
scenarios:
  slow: {v: 1 m/s}
  fast: {v: 4 m/s}
run: {t_end: 20 ms}
"""


def test_budget_outputs():
    o = load_system(M).run()
    tot = (0.8 + 0.15 + 0.45 + 0.025) / 0.45359237
    assert o["mass_total"] == pytest.approx(tot, rel=1e-6)
    assert o["mass_margin"] == pytest.approx(3 - tot, rel=1e-6)
    c = Api().check(M)
    assert c["budget"]["unit"] == "lb" and len(c["budget"]["items"]) == 4


def test_scenarios_side_by_side():
    r = Api().scenarios_run(M)
    assert [x["name"] for x in r["rows"]] == ["as written", "slow", "fast"]
    F = {x["name"]: x["outputs"]["F"] for x in r["rows"]}
    assert F["fast"] > F["as written"] > F["slow"]


def test_diff_models(tmp_path):
    p = tmp_path / "b.yaml"; p.write_text(M.replace("v: 2 m/s", "v: 3 m/s").replace("k: 1e6", "k: 2e6"))
    r = Api().diff_models(M, str(p))
    keys = {x["key"] for x in r["changes"]}
    assert "params.v" in keys and "elements.wall.k" in keys
    assert any(x["key"] == "F" and x["other"] > x["now"] for x in r["outputs"])

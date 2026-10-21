"""The replay viewer's scene (simblocks/viz/anim.py) for the Examples-menu models (2026-10-20 review): what is drawn
as a link, what as a load arrow, how a car drives along a road, how a turning body's links stay put, shapes, beams,
planets, and Animate's watchdog."""
import os

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.viz.anim import scene

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def _scene(rel, frames=120, **kw):
    path = os.path.join(EX, rel)
    old = os.environ.get("SIMBLOCKS_MODEL_DIR")
    os.environ["SIMBLOCKS_MODEL_DIR"] = os.path.dirname(os.path.abspath(path))
    try:
        return scene(load_system(open(path).read()), frames=frames, **kw)
    finally:
        if old is None:
            os.environ.pop("SIMBLOCKS_MODEL_DIR", None)
        else:
            os.environ["SIMBLOCKS_MODEL_DIR"] = old


def test_cruise_car_drives_along_a_road_with_load_arrows():
    S = _scene("learn/1d_5_cruise_control.yaml")
    assert S["road"] and not S["links"]                        # drag, rolling, engine and grade are not springs to a wall
    names = {a["name"]: a for a in S["arrows"]}
    assert set(names) == {"air", "tyres", "engine", "hill"}
    k = len(S["t"]) // 2
    assert names["engine"]["v"][k][0] > 0 and names["air"]["v"][k][0] < 0 and names["tyres"]["v"][k][0] < 0
    assert names["hill"]["v"][-1][0] < 0                       # pulls back once the road tips up
    road = [p for b in S["bodies"] if b["ground"] for p in b["prims"] if p["kind"] == "road"]
    car = S["bodies"][0]["pos"]
    assert road and car[-1][0] - car[0][0] == pytest.approx(30.0, rel=1e-6)   # the trip spans about 30 lane widths
    assert not any(p["kind"] == "wall" for b in S["bodies"] for p in b["prims"])


def test_circuit_motors_are_arrows_not_lines():
    S = _scene("learn/1d_4_flywheel_circuit.yaml")
    assert S["links"] == []
    assert sorted(a["name"] for a in S["arrows"]) == ["bearings", "m1", "m2"] and all(a["spin"] for a in S["arrows"])


def test_links_on_turning_bodies_attach_at_fixed_points():
    S = _scene("learn/1d_3_motor_flywheel.yaml")
    gi = next(i for i, b in enumerate(S["bodies"]) if b["ground"])
    (shaft,) = S["links"]
    assert shaft["a"] == gi and shaft["b"] == gi               # world-fixed ends: they do not whip round with the spin


def test_shapes_beams_and_thrust_are_drawn():
    S = _scene("learn/3d_2_drone_arm.yaml", frames=300)
    beams = [L for L in S["links"] if L["kind"] == "beam"]
    assert len(beams) == 6 and all(L["r"] > 0 for L in beams)
    assert not any(L["name"] == "arm" and L["kind"] == "spring" for L in S["links"])   # the beam's probe is not drawn
    motor = next(b for b in S["bodies"] if b["name"] == "motor")
    assert motor["prims"][0]["kind"] == "cylinder"
    assert all(b.get("hidden") for b in S["bodies"] if "__" in b["name"])
    (thrust,) = S["arrows"]
    assert max(v[2] for v in thrust["v"]) == pytest.approx(8.0, rel=1e-3)        # up, at the CSV's peak
    assert S.get("magnified", 1) > 1 and any("bigger" in n for n in S["notes"])
    top = _scene("learn/3d_1_spinning_top.yaml")
    assert top["bodies"][0]["prims"][0]["kind"] == "cylinder"
    arm = _scene("learn/3d_3_robot_arm.yaml")
    assert sum(p["kind"] == "block" for b in arm["bodies"] for p in b["prims"]) == 2
    assert "magnified" not in arm


def test_2d_rod_lies_along_its_points_and_planets_are_drawn():
    S = _scene("learn/2d_1_pendulum.yaml")
    rod = S["bodies"][0]["prims"][0]
    assert rod["kind"] == "block" and abs(abs(rod.get("turn", 0.0)) - np.pi / 2) < 0.2
    ap = _scene("showcase/apollo11/apollo11_1_ascent.yaml", frames=60)
    pl = [p for b in ap["bodies"] for p in b["prims"] if p["kind"] == "planet"]
    assert pl and pl[0]["r"] == pytest.approx(6.371e6, rel=0.01)


def test_animate_watchdog_is_generous_and_stops_a_runaway(monkeypatch):
    from simblocks.app import api as A
    M = load_system(open(os.path.join(EX, "learn/1d_1_mass_spring_damper.yaml")).read()).nominal
    assert A.anim_watchdog(M) >= 120.0
    text = open(os.path.join(EX, "learn/1d_1_mass_spring_damper.yaml")).read()
    import re
    text = re.sub(r"t_end: [^\n,}]+", "t_end: 5000 s", text, count=1)    # far longer than the 0.2 s watchdog allows
    S = scene(load_system(text), frames=100, timeout=0.2)
    assert S["timed_out"] and 0 < S["t_stopped"] < 5000 and len(S["t"]) >= 2


def test_fast_spinners_are_split_and_frames_follow_fast_turns():
    from simblocks.viz.anim import _max_turn_deg, MAX_STEP_DEG
    top = _scene("learn/3d_1_spinning_top.yaml", frames=900)
    sp = top["bodies"][0]["spin"]
    assert sp["axis"] == [0.0, 0.0, 1.0] and max(sp["w"]) == pytest.approx(300, rel=0.02)
    assert _max_turn_deg(top) <= MAX_STEP_DEG * 1.1 and len(top["t"]) > 900     # more frames for the fast precession
    assert top["signals"][0]["label"] == "top spin" and top["signals"][0]["unit"] == "rpm"
    fly = _scene("learn/1d_4_flywheel_circuit.yaml")
    assert "spin" in fly["bodies"][0]
    slow = _scene("learn/2d_1_pendulum.yaml")
    assert "spin" not in slow["bodies"][0]                     # a slow swing keeps its exact orientations

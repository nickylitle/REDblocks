"""The bullwhip showcase (2026-10-23): 3D air drag (aero_drag on a body, air_drag on a beam), checked against closed
forms, and the tapered whip itself: the tip passes the speed of sound in air, and goes faster still in vacuum."""
import math
import os

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

HERE = os.path.join(os.path.dirname(__file__), "..", "examples", "showcase", "bullwhip")
WHIP = os.path.join(HERE, "bullwhip_crack.yaml")

BALL = """simblocks: 1
name: fall
dim: 3d
gravity: %(g)s
bodies:
  ball: {dof: free, mass: 0.1, inertia: 1e-5, at: [0, 0, 0], v0: %(v0)s}
elements:
  air: {type: aero_drag, measure: ball, area: 0.01, Cd: %(cd)s %(extra)s}
outputs:
  vz: {body: ball, quantity: vz}
  vx: {body: ball, quantity: vx}
  vy: {body: ball, quantity: vy}
  F: {element: air, quantity: force}
  q: {element: air, quantity: dynamic_pressure}
  M: {element: air, quantity: mach}
run: {t_end: %(T)s, dt: %(dt)s}
"""


def _ball(**kw):
    d = dict(g=9.81, v0="[0, 0, 0]", cd=0.5, extra="", T=20, dt=1e-4); d.update(kw)
    return load_system(BALL % d).run()


def test_bluff_drag_reaches_terminal_speed():
    o = _ball()
    vt = math.sqrt(0.1 * 9.81 / (0.5 * 1.225 * 0.5 * 0.01))
    assert o["vz"] == pytest.approx(-vt, rel=1e-6) and o["F"] == pytest.approx(0.981, rel=1e-6)
    assert o["q"] == pytest.approx(0.5 * 1.225 * vt ** 2, rel=1e-5) and o["M"] == pytest.approx(vt / 343, rel=1e-5)
    assert o["_ledger_rel"] < 1e-4


def test_coasting_against_quadratic_drag():
    o = _ball(g=0, v0="[100, 0, 0]", T=0.5, dt=1e-5)
    k = 0.5 * 1.225 * 0.5 * 0.01 / 0.1                       # v' = -k v^2: v = v0 / (1 + k v0 t)
    assert o["vx"] == pytest.approx(100 / (1 + k * 100 * 0.5), rel=2e-5)
    assert o["_ledger_rel"] < 1e-4


def test_slender_part_feels_crossflow_only():
    along = _ball(g=0, v0="[100, 0, 0]", cd=1.2, extra=", axis: x", T=0.2, dt=1e-5)
    assert along["vx"] == pytest.approx(100.0, abs=1e-9)        # moving along its axis: no skin drag given
    skin = _ball(g=0, v0="[100, 0, 0]", cd=1.2, extra=", axis: x, Cd_axial: 0.01, area_axial: 0.03", T=0.5, dt=1e-5)
    k = 0.5 * 1.225 * 0.01 * 0.03 / 0.1
    assert skin["vx"] == pytest.approx(100 / (1 + k * 100 * 0.5), rel=2e-5)
    cross = _ball(g=0, v0="[0, 100, 0]", cd=1.2, extra=", axis: x", T=0.5, dt=1e-5)
    k = 0.5 * 1.225 * 1.2 * 0.01 / 0.1
    assert cross["vy"] == pytest.approx(100 / (1 + k * 100 * 0.5), rel=1e-4)


def test_cd_against_mach_and_the_off_switch():
    o = _ball(g=0, v0="[400, 0, 0]", cd="{mach: [0, 1, 2], cd: [1, 2, 2]}", T=1e-4, dt=1e-6)
    assert o["F"] == pytest.approx(0.5 * 1.225 * 0.01 * 2.0 * o["vx"] ** 2, rel=1e-3)   # past Mach 1: Cd 2
    off = _ball(g=0, v0="[400, 0, 0]", extra=", scale: 0", T=0.1, dt=1e-5)
    assert off["vx"] == 400.0 and off["F"] == 0.0
    with pytest.raises(SpecError, match="go with axis"):
        _ball(extra=", Cd_axial: 0.01")
    with pytest.raises(SpecError, match="one moving body|not in any element"):
        load_system((BALL % dict(g=0, v0="[0,0,0]", cd=1, extra="", T=1, dt=1e-3)).replace("measure: ball", "measure: nope")).compile()


def test_beam_air_drag_puts_drag_on_every_node():
    doc = """simblocks: 1
name: rope in wind
dim: 3d
gravity: 0
bodies:
  a: {dof: free, mass: 1e-6, inertia: 1e-12, at: [0, 0, 0], points: {c: [0, 0, 0]}, v0: [0, 0, 50]}
  b: {dof: free, mass: 1e-6, inertia: 1e-12, at: [1, 0, 0], points: {c: [0, 0, 0]}, v0: [0, 0, 50]}
elements:
  rod: {type: beam, measure: [a.c, b.c], section: {block: circle, d: 4 mm}, E: 2 GPa, G: 0.75 GPa, density: 1150,
        segments: 5, orient: y, air_drag: {Cd: 1.2, scale: %s}}
outputs:
  vz: {body: b, quantity: vz}
run: {t_end: 0.01, dt: 2e-6}
"""
    S = load_system(doc % 1); M = S.nominal
    air = [n for n in M.el_names if "__air" in n]
    assert len(air) == 6                                    # 4 nodes and both ends
    m = 1150 * math.pi / 4 * 0.004 ** 2 * 1.0 + 2e-6            # a rod broadside, all moving together
    o = S.run(); k = 0.5 * 1.225 * 1.2 * 0.004 * 1.0 / m
    assert o["vz"] == pytest.approx(50 / (1 + k * 50 * 0.01), rel=2e-3)
    assert load_system(doc % 0).run()["vz"] == pytest.approx(50.0, rel=1e-9)


def test_rod_shape_has_a_round_bar_polar_inertia():
    from simblocks.rigid_body.inertia import mass_properties
    g = mass_properties("rod", mass=1.0, L=0.5, A=1e-4)
    assert g["Izz"] == pytest.approx(1e-4 / (2 * math.pi)) and g["Ixx"] == pytest.approx(0.5 ** 2 / 12)


@pytest.fixture(scope="module")
def whip():
    return load_system(open(WHIP).read())


def test_the_whip_cracks(whip):
    o = whip.run()
    assert 1.3 < o["mach"] < 1.5                          # supersonic, near the measured Mach 1-2 of real whips
    assert 0.12 < o["t_crack"] / 1e3 < 0.15 and 15 < o["hand_speed"] < 25
    assert o["_ledger_rel"] < 1e-3 and o["_beam_bend_max"] < 0.5   # energy closes, every segment short enough


def test_finer_mesh_and_vacuum(tmp_path):
    import subprocess, sys
    fine = tmp_path / "fine.yaml"
    subprocess.run([sys.executable, os.path.join(HERE, "build_whip.py"), "--segments", "14", "--out", str(fine)], check=True)
    base = load_system(open(WHIP).read()).run()
    o = load_system(fine.read_text()).run()
    assert o["tip_speed"] == pytest.approx(base["tip_speed"], rel=0.03)
    txt = open(WHIP).read().replace("air: {value: 1,", "air: {value: 0,").replace("t_stop: {value: 0.175 s,", "t_stop: {value: 0.136 s,")
    v = load_system(txt).run()
    assert v["tip_speed"] > 1.2 * base["tip_speed"]          # no air: the loop keeps its energy, the tip goes faster


def test_replay_shows_the_whip_its_tip_and_the_boom(whip):
    from simblocks.viz.anim import scene
    S = scene(whip, frames=400)
    assert "magnified" not in S                            # a whip bends far: nothing is drawn bigger
    trails = [b["name"] for b in S["bodies"] if b.get("pos") and b.get("trail") is not False]
    assert trails == ["cracker"] and max(next(b for b in S["bodies"] if b["name"] == "cracker")["trail_speed"]) > 400
    (bm,) = S["booms"]
    assert bm["body"] == "cracker" and bm["c"] == 343.0 and 0.125 < bm["t"] < 0.14
    assert S["signals"][0]["key"] == "control.tip_mach" and S["signals"][0]["show"]

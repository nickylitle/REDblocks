"""Rigid impact contact (impulses, restitution e, Coulomb mu): every joule lost is booked as heat."""
import numpy as np
import pytest

from simblocks.compiler.system import load_system


@pytest.mark.parametrize("e", [0.0, 0.5, 1.0])
def test_1d_two_body_collision(e):
    """m1 = 1 at 3 m/s hits m2 = 2 at rest: Newton's law of restitution; momentum kept; heat = 1/2 mu v^2 (1-e^2)."""
    S = load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  a: {{dof: translation, mass: 1, v0: 3}}
  b: {{dof: translation, mass: 2, x0: 0.1}}
elements:
  hit: {{type: impact, e: {e}, measure: a - b}}
outputs:
  va: {{body: a, quantity: velocity}}
  vb: {{body: b, quantity: velocity}}
  heat: {{element: hit, quantity: heat}}
  J: {{element: hit, quantity: impulse, reduce: max}}
run: {{t_end: 0.1 s, dt: 1e-5 s}}
""")
    o = S.run()
    m1, m2, v = 1.0, 2.0, 3.0
    va = (m1 * v - m2 * e * v) / (m1 + m2); vb = (m1 * v + m1 * e * v) / (m1 + m2)
    assert o["va"] == pytest.approx(va, abs=1e-9) and o["vb"] == pytest.approx(vb, abs=1e-9)
    mu = m1 * m2 / (m1 + m2)
    assert o["heat"] == pytest.approx(0.5 * mu * v * v * (1 - e * e), abs=1e-9)
    assert o["J"] == pytest.approx(mu * v * (1 + e), rel=1e-9)
    assert o["_ledger_rel"] < 1e-9


def _drop(dim, e, t_end=0.9):
    if dim == "2d":
        bodies = """  floor: {ground: true, lines: {top: {from: [-1, 0], to: [1, 0]}}}
  ball: {dof: planar, mass: 0.1, inertia: 1e-5, at: [0, 1.0], circles: {rim: {at: [0, 0], r: 0.02}}}"""
        meas = "{contact: [ball.rim, floor.top]}"; q = "y"
    else:
        bodies = """  floor: {ground: true, planes: {top: {at: [0, 0, 0], normal: [0, 0, 1], size: [4, 4]}}}
  ball: {dof: free, mass: 0.1, inertia: 1.6e-5, at: [0, 0, 1.0], spheres: {rim: {at: [0, 0, 0], r: 0.02}}}"""
        meas = "{contact: [ball.rim, floor.top]}"; q = "z"
    return load_system(f"""simblocks: 1
name: t
dim: {dim}
gravity: 9.81
bodies:
{bodies}
elements:
  hit: {{type: impact, e: {e}, measure: {meas}}}
outputs:
  heat: {{element: hit, quantity: heat}}
run: {{t_end: {t_end} s, dt: 1e-4 s}}
"""), q


@pytest.mark.parametrize("dim", ["2d", "3d"])
@pytest.mark.parametrize("e", [0.6, 1.0])
def test_drop_rebound_height_and_heat(dim, e):
    S, q = _drop(dim, e)
    o, H = S.run(history=True, n_log=9000)
    z = H[f"ball.{q}"]; i = int(np.argmin(z))
    assert z.min() == pytest.approx(0.02, abs=5e-4)                   # never sinks into the floor
    assert z[i:].max() == pytest.approx(0.02 + e * e * 0.98, abs=2e-3)
    assert o["heat"] == pytest.approx(0.1 * 9.81 * 0.98 * (1 - e * e), abs=2e-3)
    assert o["_ledger_rel"] < 1e-3


def test_heat_history_is_cumulative():
    S, q = _drop("2d", 0.5, t_end=1.5)
    o, H = S.run(history=True, n_log=3000)
    h = H["hit.heat"]
    assert np.all(np.diff(h) >= -1e-12) and h[-1] == pytest.approx(o["heat"])


def test_3d_sliding_ball_starts_rolling():
    """Ball launched sliding at v0 with friction: rolls at 5/7 v0; heat = 1/2 m v0^2 (2/7)."""
    S = load_system("""simblocks: 1
name: t
dim: 3d
gravity: 9.81
bodies:
  floor: {ground: true, planes: {top: {at: [0, 0, 0], normal: [0, 0, 1], size: [40, 4]}}}
  ball: {dof: free, mass: 0.1, inertia: 1.6e-5, at: [0, 0, 0.02], v0: [2, 0, 0], spheres: {s: {at: [0, 0, 0], r: 0.02}}}
elements:
  hit: {type: impact, e: 0, mu: 0.3, measure: {contact: [ball.s, floor.top]}}
outputs:
  vx: {body: ball, quantity: vx}
  wy: {body: ball, quantity: wy}
  heat: {element: hit, quantity: heat}
run: {t_end: 1.0 s, dt: 1e-4 s}
""")
    o = S.run()
    assert o["vx"] == pytest.approx(2 * 5 / 7, rel=1e-6)
    assert o["wy"] == pytest.approx(2 * 5 / 7 / 0.02, rel=1e-6)
    assert o["heat"] == pytest.approx(0.5 * 0.1 * 4 * 2 / 7, rel=0.02)


def test_2d_oblique_disc_hits_disc():
    """Equal discs, frictionless, e = 1, glancing hit: momentum and energy kept, no heat."""
    S = load_system("""simblocks: 1
name: t
dim: 2d
bodies:
  a: {dof: planar, mass: 1, inertia: 1e-3, at: [0, 0], v0: [2, 0], circles: {c: {at: [0, 0], r: 0.05}}}
  b: {dof: planar, mass: 1, inertia: 1e-3, at: [0.3, 0.05], circles: {c: {at: [0, 0], r: 0.05}}}
elements:
  hit: {type: impact, e: 1, measure: {contact: [a.c, b.c]}}
outputs:
  vax: {body: a, quantity: vx}
  vay: {body: a, quantity: vy}
  vbx: {body: b, quantity: vx}
  vby: {body: b, quantity: vy}
  heat: {element: hit, quantity: heat}
run: {t_end: 0.3 s, dt: 1e-5 s}
""")
    o = S.run()
    assert o["vax"] + o["vbx"] == pytest.approx(2.0, abs=1e-9) and o["vay"] + o["vby"] == pytest.approx(0, abs=1e-9)
    ke = 0.5 * (o["vax"] ** 2 + o["vay"] ** 2 + o["vbx"] ** 2 + o["vby"] ** 2)
    assert ke == pytest.approx(2.0, rel=1e-9) and abs(o["heat"]) < 1e-9
    # line of centres at contact: 30 deg from the path (offset 0.05 over radii 0.1): b leaves along it
    assert np.degrees(np.arctan2(o["vby"], o["vbx"])) == pytest.approx(30.0, abs=0.5)

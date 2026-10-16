"""Rigid (projected) joints: exact constraints, no stiffness to choose; real losses booked, drift reported."""
import math

import numpy as np
import pytest

from simblocks.compiler.system import load_system

L = 0.5; m = 1.0; I = m * L * L / 12; a0 = 0.1; T_EXP = 2 * math.pi * math.sqrt(2 * L / (3 * 9.81))


def _period(t, x):
    zc = [t[i] for i in range(1, len(x)) if x[i - 1] > 0 and x[i] <= 0]
    return float(np.mean(np.diff(zc)))


def test_2d_rigid_hinge_pendulum():
    S = load_system(f"""simblocks: 1
name: t
dim: 2d
gravity: 9.81
bodies:
  wall: {{ground: true, points: {{o: [0, 0]}}}}
  bar: {{dof: planar, mass: {m}, inertia: {I}, at: [{L / 2 * math.sin(a0)}, {-L / 2 * math.cos(a0)}], angle0: {a0},
         points: {{top: [0, {L / 2}]}}}}
elements:
  pivot: {{type: hinge, rigid: true, measure: [wall.o, bar.top]}}
outputs:
  drift: {{element: pivot, quantity: drift, reduce: max}}
run: {{t_end: 3 s, dt: 1e-4 s}}
""")
    o, H = S.run(history=True, n_log=30000)
    assert _period(H["t"], H["bar.angle"]) == pytest.approx(T_EXP, rel=2e-3)
    assert o["_ledger_rel"] < 2e-3 and o["drift"] < 1e-6


def test_3d_rigid_hinge_blocks_other_axes():
    """Pendulum on a rigid y hinge, started with a spin about x as well: the hinge removes the x spin at once
    (booked as a real loss, 1/2 I_eff w^2) and the swing period is the compound pendulum's."""
    S = load_system(f"""simblocks: 1
name: t
dim: 3d
gravity: 9.81
bodies:
  wall: {{ground: true, points: {{o: [0, 0, 0]}}}}
  bar: {{dof: free, mass: {m}, inertia: [{I}, {I}, 1e-4], at: [{L / 2 * math.sin(a0)}, 0, {-L / 2 * math.cos(a0)}],
         orient0: {{axis: [0, 1, 0], angle: {-a0}}}, points: {{top: [0, 0, {L / 2}]}}}}
elements:
  pivot: {{type: hinge, rigid: true, axis: y, measure: [wall.o, bar.top]}}
outputs:
  ymax: {{body: bar, quantity: y, reduce: max_abs}}
  wx: {{body: bar, quantity: wx, reduce: max_abs}}
run: {{t_end: 3 s, dt: 1e-4 s}}
""")
    o, H = S.run(history=True, n_log=30000)
    assert _period(H["t"], H["bar.x"]) == pytest.approx(T_EXP, rel=2e-3)
    assert o["ymax"] < 1e-9 and o["_ledger_rel"] < 2e-3


def test_rigid_pin_catch_books_the_loss():
    """1D-like catch: a free disc moving at v is pinned to a fixed point at t = 0: all its speed along the pin
    goes; the joint books 1/2 m v^2 exactly as dissipated."""
    S = load_system("""simblocks: 1
name: t
dim: 2d
bodies:
  wall: {ground: true, points: {o: [0, 0]}}
  puck: {dof: translation, mass: 2, at: [0, 0], v0: [3, 0], points: {c: [0, 0]}}
elements:
  catch: {type: pin, rigid: true, measure: [wall.o, puck.c]}
outputs:
  heat: {element: catch, quantity: dissipated}
  vx: {body: puck, quantity: vx}
run: {t_end: 0.01 s, dt: 1e-5 s}
""")
    o = S.run()
    assert abs(o["vx"]) < 1e-12 and o["heat"] == pytest.approx(0.5 * 2 * 9, rel=1e-9)
    assert o["_ledger_rel"] < 1e-9


def test_rigid_weld_two_halves_fly_as_one():
    """Two half bars welded rigidly, thrown spinning in free space: they keep their relative pose exactly and
    the pair spins at the combined body's rate (angular momentum / combined inertia)."""
    S = load_system("""simblocks: 1
name: t
dim: 2d
bodies:
  a: {dof: planar, mass: 1, inertia: 0.001, at: [-0.05, 0], omega0: 10, v0: [0, -0.5], points: {j: [0.05, 0]}}
  b: {dof: planar, mass: 1, inertia: 0.001, at: [0.05, 0], omega0: 10, v0: [0, 0.5], points: {j: [-0.05, 0]}}
elements:
  w: {type: weld, rigid: true, measure: [a.j, b.j]}
outputs:
  wa: {body: a, quantity: omega}
  wb: {body: b, quantity: omega}
  drift: {element: w, quantity: drift, reduce: max}
run: {t_end: 1 s, dt: 1e-4 s}
""")
    o = S.run()
    # rigid pair: I = 2 * 0.001 + 2 * (1 * 0.05^2) = 0.007; started consistent (v = w r), so w stays 10
    assert o["wa"] == pytest.approx(10.0, rel=1e-6) and o["wb"] == pytest.approx(10.0, rel=1e-6)
    assert o["drift"] < 1e-6 and o["_ledger_rel"] < 1e-3


def test_rigid_slider_is_a_frictionless_guide():
    """Block on a rigid slider tilted 30 deg under gravity: a = g sin 30 along the guide, none across it."""
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    S = load_system(f"""simblocks: 1
name: t
dim: 2d
gravity: 9.81
bodies:
  rail: {{ground: true, points: {{o: [0, 0]}}}}
  blk: {{dof: planar, mass: 1, inertia: 1e-3, at: [0, 0], points: {{p: [0, 0]}}}}
elements:
  guide: {{type: slider, rigid: true, axis: [{c}, {-s}], k_rot: 0, measure: [rail.o, blk.p]}}
outputs:
  slid: {{element: guide, quantity: measure}}
  ang: {{body: blk, quantity: angle, reduce: max_abs}}
run: {{t_end: 0.5 s, dt: 1e-4 s}}
""")
    o = S.run()
    assert o["slid"] == pytest.approx(0.5 * 9.81 * s * 0.25, rel=2e-3)
    assert o["ang"] < 1e-9

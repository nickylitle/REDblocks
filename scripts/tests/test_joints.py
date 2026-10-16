"""2D / 3D joints and the twist measure."""
import math

import pytest

from simblocks.compiler.system import load_system


def run(dim, body, **kw):
    S = load_system(f"simblocks: 1\nname: t\ndim: {dim}\n" + body)
    return S, S.run(**kw)


def test_rope_only_pulls():
    """A ball on a slack rope falls freely until the rope goes taut."""
    S, o = run("2d", """gravity: 9.81 m/s^2
bodies:
  top: {ground: true, points: {hook: [0, 0]}}
  ball: {dof: [x, y], mass: 1 kg, at: [0, -0.5 m], points: {c: [0, 0]}}
elements:
  rope: {type: rope, k: 1e5 N/m, c: 20, measure: {distance: [top.hook, ball.c], length: 1 m}}
outputs:
  t_taut: {element: rope, quantity: force, reduce: first_time_above, value: 1 N}
  F: {element: rope, quantity: force, reduce: min}
run: {t_end: 0.6 s}
""")
    assert o["t_taut"] == pytest.approx(math.sqrt(2 * 0.5 / 9.81), rel=0.01)
    assert o["F"] >= 0                                   # never pushes


def test_weld_2d_makes_one_rigid_body():
    """Two bars welded end to end swing as one compound pendulum."""
    S, o = run("2d", """gravity: 9.81 m/s^2
bodies:
  g: {ground: true, points: {o: [0, 0]}}
  a: {mass: 1 kg, inertia: "= 1*0.2**2/12", at: [0, 0], com: [0.1 m, 0], angle0: -80 deg, points: {root: [0, 0], tip: [0.2 m, 0]}}
  b: {mass: 1 kg, inertia: "= 1*0.2**2/12", at: [0.03473 m, -0.19696 m], com: [0.1 m, 0], angle0: -80 deg, points: {root: [0, 0]}}
elements:
  pivot: {type: pin, k: 1e8, c: 200, measure: [g.o, a.root]}
  joint: {type: weld, k: 1e8, c: 200, k_rot: 1e5, c_rot: 2, measure: [a.tip, b.root]}
outputs:
  t: {body: a, quantity: angle, reduce: first_time_below, value: -90 deg}
run: {t_end: 1 s}
""")
    L = 0.4; T = 2 * math.pi * math.sqrt(2 * L / (3 * 9.81))
    assert o["t"] == pytest.approx(T / 4, rel=0.01)


def test_slider_2d_guides_and_reports_slide():
    """A block on a 30 degree slider (no friction) slides with a = g sin(30) and reports how far it went."""
    S, o = run("2d", """gravity: 9.81 m/s^2
bodies:
  rail: {ground: true, points: {o: [0, 0]}}
  block: {dof: [x, y], mass: 1 kg, at: [0, 0], points: {c: [0, 0]}}
elements:
  guide: {type: slider, k: 1e6, c: 100, axis: [0.8660254, -0.5], measure: [rail.o, block.c]}
outputs:
  slid: {element: guide, quantity: measure}
  v: {body: block, quantity: speed}
run: {t_end: 0.5 s}
""")
    a = 9.81 * 0.5
    assert o["slid"] == pytest.approx(0.5 * a * 0.25, rel=0.01)
    assert o["v"] == pytest.approx(a * 0.5, rel=0.01)


def test_twist_drive_between_free_bodies_3d_reaction():
    """A motor between a free chassis and a free rotor: equal and opposite torques, angular momentum stays zero."""
    S, o = run("3d", """bodies:
  chassis: {dof: [rotate], inertia: [0.02, 0.02, 0.01], points: {m: [0, 0, 0]}}
  rotor: {dof: [rotate], inertia: [0.001, 0.001, 0.002], points: {m: [0, 0, 0]}}
elements:
  motor: {type: force, P: 0.5, measure: {twist: [chassis, rotor], axis: z}}
outputs:
  wr: {body: rotor, quantity: wz}
  wc: {body: chassis, quantity: wz}
run: {t_end: 0.2 s}
""")
    assert o["wr"] == pytest.approx(0.5 * 0.2 / 0.002, rel=1e-3)
    assert 0.002 * o["wr"] + 0.01 * o["wc"] == pytest.approx(0.0, abs=1e-9)


def test_hinge_3d_free_about_axis_locked_across():
    """A disc on a 3d hinge (axis z) spun by a twist torque stays on its axis; the hinge resists tilting."""
    S, o = run("3d", """bodies:
  frame: {ground: true, points: {o: [0, 0, 0]}}
  disc: {mass: 1 kg, inertia: [0.002, 0.002, 0.004], points: {c: [0, 0, 0]}, omega0: [0.5, 0, 0]}
elements:
  axle: {type: hinge, k: 1e7, c: 100, axis: z, k_rot: 500, c_rot: 2, measure: [frame.o, disc.c]}
  spin: {type: force, P: 0.4, measure: {twist: [frame, disc], axis: z}}
outputs:
  wz: {body: disc, quantity: wz}
  wx: {body: disc, quantity: wx, reduce: max_abs}
  tilt_final: {body: disc, quantity: wx}
run: {t_end: 0.5 s}
""")
    assert o["wz"] == pytest.approx(0.4 * 0.5 / 0.004, rel=0.01)
    assert abs(o["tilt_final"]) < 0.5                     # the initial wobble is held by the hinge, not let run


def test_weld_3d_rigid_pair_rotates_together():
    """Two free bodies welded together: a torque on one (from ground) spins both at the combined inertia."""
    S, o = run("3d", """bodies:
  g: {ground: true}
  a: {mass: 1 kg, inertia: 0.001, points: {p: [0, 0, 0]}}
  b: {mass: 1 kg, inertia: 0.001, points: {p: [0, 0, 0]}}
elements:
  w: {type: weld, k: 1e7, c: 50, k_rot: 1e4, c_rot: 2, measure: [a.p, b.p]}
  t: {type: force, P: 0.02, measure: {twist: [g, a], axis: z}}
outputs:
  wa: {body: a, quantity: wz}
  wb: {body: b, quantity: wz}
run: {t_end: 0.2 s}
""")
    assert o["wa"] == pytest.approx(0.02 * 0.2 / 0.002, rel=0.01)     # a force pushes its measure positive
    assert o["wb"] == pytest.approx(o["wa"], rel=0.01)

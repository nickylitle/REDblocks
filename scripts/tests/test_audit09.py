"""Regression tests for the v0.9 solver audit: each case reproduced a defect before its fix."""
import math

import pytest

from simblocks.compiler.loader import SpecError
from simblocks.compiler.system import load_system
from simblocks.materials import get


def run(txt, **kw):
    return load_system(txt).run(**kw)


def test_ledger_static_weld_is_not_100_percent():
    o = run("""simblocks: 1
name: t
dim: 2d
gravity: 9.81
bodies:
  wall: {ground: true, points: {o: [0, 0]}}
  arm: {dof: planar, mass: 1, inertia: 0.01, at: [0.3, 0], points: {root: [-0.3, 0]}}
elements:
  j: {type: weld, rigid: true, measure: [wall.o, arm.root]}
outputs:
  F: {element: j, quantity: force}
run: {t_end: 0.2 s, dt: 0.1 ms}
""")
    assert o["_ledger_rel"] < 1e-3
    assert o["F"] == pytest.approx(9.81, rel=1e-3)


def test_ledger_uses_peak_energy_for_a_pendulum_ending_at_rest():
    L = 0.5
    T = 2 * math.pi * math.sqrt(L / 9.81)
    o = run(f"""simblocks: 1
name: t
dim: 2d
gravity: 9.81
bodies:
  wall: {{ground: true, points: {{o: [0, 0]}}}}
  bob: {{dof: planar, mass: 1, inertia: 1e-4, at: [{L * math.sin(0.3)}, {-L * math.cos(0.3)}], points: {{c: [0, 0]}}}}
elements:
  rod: {{type: kelvin_voigt, k: 1e7, c: 0, measure: {{distance: [wall.o, bob.c]}}}}
outputs:
  x: {{body: bob, quantity: x}}
run: {{t_end: {3 * T} s, dt: 20 us}}
""")
    assert o["_ledger_rel"] < 0.02


def _hinge_case(axis, P):
    L = 0.3
    return run(f"""simblocks: 1
name: t
dim: 3d
bodies:
  wall: {{ground: true, points: {{root: [0, 0, 0]}}}}
  tip: {{dof: free, mass: 0.05 kg, inertia: 2e-6, at: [{L}, 0, 0], points: {{p: [0, 0, 0]}}}}
elements:
  bar: {{type: beam, E: 200 GPa, density: 7850 kg/m^3, section: {{block: rectangle, b: 20 mm, h: 4 mm}},
         segments: 4, damping: 0.3, orient: y, hinge: auto, material: 4130_normalized, measure: [wall.root, tip.p]}}
  push: {{type: force, P: {P}, measure: tip.{axis}}}
  calm: {{type: dashpot, c: 30, measure: tip.{axis}}}
outputs:
  hs: {{element: bar, quantity: hinge_set, reduce: max}}
  u: {{body: tip, quantity: {axis}}}
run: {{t_end: 0.6 s}}
""")


def test_3d_hinge_strong_axis_uses_its_own_plastic_moment():
    sy = get("4130_normalized").sy; b = 0.02; h = 0.004; L = 0.3
    Mp_strong = sy * h * b * b / 4
    o = _hinge_case("z", -0.5 * Mp_strong / L)
    assert o["hs"] == pytest.approx(0.0, abs=1e-9)
    assert o["u"] == pytest.approx(-0.5 * Mp_strong / L * L ** 3 / (3 * 200e9 * h * b ** 3 / 12), rel=0.03)
    assert o["_ledger_rel"] < 1e-3


@pytest.mark.parametrize("e,mu", [(0.0, 0.5), (0.5, 0.0), (0.5, 0.5)])
def test_resting_plate_books_nothing_and_does_not_chatter(e, mu):
    o = run(f"""simblocks: 1
name: t
dim: 2d
gravity: 9.81
bodies:
  floor: {{ground: true, lines: {{top: {{from: [-1, 0], to: [1, 0]}}}}}}
  plate: {{dof: planar, mass: 1, inertia: 0.001, at: [0, 0.02], points: {{l: [-0.05, -0.02], r: [0.05, -0.02]}}}}
elements:
  hl: {{type: impact, e: {e}, mu: {mu}, measure: {{contact: [plate.l, floor.top]}}}}
  hr: {{type: impact, e: {e}, mu: {mu}, measure: {{contact: [plate.r, floor.top]}}}}
outputs:
  dl: {{element: hl, quantity: dissipated}}
  dr: {{element: hr, quantity: dissipated}}
  Fl: {{element: hl, quantity: force}}
  y: {{body: plate, quantity: y}}
run: {{t_end: 2 s, dt: 0.1 ms}}
""")
    assert o["dl"] + o["dr"] < 1e-5
    assert o["Fl"] == pytest.approx(4.905, rel=1e-3)
    assert o["y"] == pytest.approx(0.02, abs=1e-6)
    assert o["_ledger_rel"] < 1e-3


def test_rigid_joint_must_start_closed():
    with pytest.raises(SpecError, match="apart at the start"):
        load_system("""simblocks: 1
name: t
dim: 2d
bodies:
  wall: {ground: true, points: {o: [0, 0]}}
  bar: {dof: planar, mass: 1, inertia: 0.01, at: [0, -0.3], points: {top: [0, 0.28]}}
elements:
  j: {type: hinge, rigid: true, measure: [wall.o, bar.top]}
outputs:
  F: {element: j, quantity: force}
run: {t_end: 0.1 s}
""")


def test_3d_beam_needs_iy():
    with pytest.raises(SpecError, match="needs Iy"):
        load_system("""simblocks: 1
name: t
dim: 3d
bodies:
  wall: {ground: true, points: {r: [0, 0, 0]}}
  tip: {dof: free, mass: 0.05, inertia: 1e-6, at: [0.3, 0, 0], points: {p: [0, 0, 0]}}
elements:
  bar: {type: beam, E: 200 GPa, G: 80 GPa, section: {block: given_section, I: 1e-10, A: 8e-5, J: 3e-10},
        measure: [wall.r, tip.p]}
outputs:
  u: {body: tip, quantity: z}
run: {t_end: 0.01 s}
""")


def test_throttle_outside_0_to_1_is_refused():
    with pytest.raises(SpecError, match="throttle"):
        load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  shaft: {dof: rotation, inertia: 1e-4}
elements:
  mot: {type: drive, kv: 1000, R: 0.05, I_max: 40, cells: 4, R0: 0.02, capacity: 1 Ah, throttle: 1.5,
        measure: shaft}
outputs:
  w: {body: shaft, quantity: velocity}
run: {t_end: 0.01 s}
""")


def test_shape_axis_is_world_direction_with_orient0():
    S = load_system("""simblocks: 1
name: t
dim: 3d
bodies:
  c: {dof: free, shape: {kind: cylinder, R: 0.05, L: 0.4, mass: 3, axis: z},
      orient0: {axis: x, angle: 90 deg}, at: [0, 0, 0]}
elements:
  f: {type: force, P: 0, measure: c.z}
outputs:
  z: {body: c, quantity: z}
run: {t_end: 0.001 s}
""")
    import numpy as np
    from simblocks.compiler.model3d import _qR
    a = S.nominal.arrays
    R = _qR(a["quat0"][0]); Ib = np.linalg.inv(a["Iinv"][0])
    Iw = R @ Ib @ R.T
    assert Iw[2, 2] == pytest.approx(0.5 * 3 * 0.05 ** 2, rel=1e-6)


def test_list_inertia_must_be_physical():
    with pytest.raises(SpecError, match="not physical"):
        load_system("""simblocks: 1
name: t
dim: 3d
bodies:
  b: {dof: free, mass: 1, inertia: [0.001, 0.001, 0.005]}
elements:
  f: {type: force, P: 0, measure: b.z}
outputs:
  z: {body: b, quantity: z}
run: {t_end: 0.001 s}
""")


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_rigid_weld_on_long_light_arm_has_no_start_spike(dim):
    g = "{ground: true, points: {o: [0, 0]}}" if dim == "2d" else "{ground: true, points: {o: [0, 0, 0]}}"
    b = ("{dof: planar, mass: 1 kg, inertia: 0.01, at: [3, 0], points: {w: [-3, 0]}}" if dim == "2d" else
         "{dof: free, mass: 1 kg, inertia: 0.01, at: [3, 0, 0], points: {w: [-3, 0, 0]}}")
    o = run(f"""simblocks: 1
name: t
dim: {dim}
gravity: 9.81
bodies:
  base: {g}
  blk: {b}
elements:
  grip: {{type: weld, rigid: true, measure: [base.o, blk.w]}}
outputs:
  Fmax: {{element: grip, quantity: force, reduce: max}}
  Mmax: {{element: grip, quantity: moment, reduce: max_abs}}
run: {{t_end: 0.2 s, dt: 0.1 ms}}
""")
    assert o["Fmax"] == pytest.approx(9.81, rel=1e-6)
    assert o["Mmax"] == pytest.approx(29.43, rel=1e-6)


def test_beam_damping_reaches_the_first_mode():
    import numpy as np
    S = load_system("""simblocks: 1
name: t
dim: 3d
bodies:
  wall: {ground: true, points: {root: [0, 0, 0]}}
  tip: {dof: free, mass: 1e-4 kg, inertia: 2e-7, at: [0, 0, 0.3], v0: [0.05, 0, 0], points: {p: [0, 0, 0]}}
elements:
  bar: {type: beam, E: 200 GPa, density: 7850 kg/m^3, section: {block: rectangle, b: 20 mm, h: 4 mm}, segments: 4,
        damping: 0.05, measure: [wall.root, tip.p]}
outputs:
  x: {body: tip, quantity: x, reduce: max_abs}
run: {t_end: 0.4 s, timeout: none}
""")
    o, H = S.run(history=True, n_log=20000)
    t = H["t"]; x = H["tip.x"]
    a1 = np.abs(x[(t > 0.0) & (t < 0.05)]).max(); a2 = np.abs(x[(t > 0.35) & (t < 0.4)]).max()
    zeta = math.log(a1 / a2) / (227.7 * 0.35)
    assert zeta == pytest.approx(0.05, rel=0.15)
    assert o["_ledger_rel"] < 0.05


def test_torque_table_on_a_translation_is_refused():
    with pytest.raises(SpecError, match="is a torque"):
        load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  m: {dof: translation, mass: 1}
elements:
  push: {type: force, measure: m, csv: {x: [0, 1], y: [5, 5], y_unit: N m}}
outputs:
  x: {body: m, quantity: position}
run: {t_end: 0.01 s}
""")


def test_propeller_thrust_falls_with_airspeed():
    from simblocks.aero.basic import propeller_thrust
    T0 = propeller_thrust(0.1, 1.2, 100, 0.25)
    assert T0 == pytest.approx(0.1 * 1.2 * 100 ** 2 * 0.25 ** 4)
    # J = 10/(100*0.25) = 0.4, J0 = 0.2/0.25 = 0.8: half the static thrust
    assert propeller_thrust(0.1, 1.2, 100, 0.25, V=10, pitch=0.2) == pytest.approx(0.5 * T0)
    assert propeller_thrust(0.1, 1.2, 100, 0.25, V=20, J0=0.8) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="pitch"):
        propeller_thrust(0.1, 1.2, 100, 0.25, V=10)


def test_expression_outputs_read_si_whatever_the_display_unit():
    txt = """simblocks: 1
name: t
dim: 1d
bodies:
  r: {dof: rotation, inertia: 1e-3, v0: 100 rad/s}
elements:
  d: {type: dashpot, c: 0, measure: r}
outputs:
  w: {body: r, quantity: velocity, unit: rpm}
  twice: {expr: "= 2 * $w"}
run: {t_end: 1 ms}
"""
    o = run(txt)
    assert o["w"] == pytest.approx(100 * 60 / (2 * math.pi), rel=1e-6)
    assert o["twice"] == pytest.approx(200.0, rel=1e-6)

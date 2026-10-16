"""2D tier: analytic checks of the planar engine and compiler."""
import math

import pytest

from simblocks.compiler.system import load_system
from simblocks.compiler.loader import SpecError

HEAD = "simblocks: 1\nname: t\ndim: 2d\n"


def run(body, **kw):
    S = load_system(HEAD + body)
    return S, S.run(**kw)


def test_pendulum_quarter_period_and_energy():
    S, o = run("""gravity: 9.81 m/s^2
bodies:
  ground: {ground: true, points: {pivot: [0, 0]}}
  bob: {mass: 1 kg, inertia: 1e-6 kg m^2, at: [0.5 m, 0], points: {c: [0, 0]}}
elements:
  rod: {type: spring, k: 1e6 N/m, measure: {distance: [ground.pivot, bob.c]}}
outputs:
  vmax: {body: bob, quantity: speed, reduce: max}
  tcross: {body: bob, quantity: x, reduce: first_time_below, value: 0}
run: {t_end: 0.6 s}
""")
    T4 = math.sqrt(0.5 / 9.81) * 1.854075             # quarter period from 90 degrees: sqrt(L/g) K(sin 45)
    assert o["tcross"] == pytest.approx(T4, rel=2e-3)
    assert o["vmax"] == pytest.approx(math.sqrt(2 * 9.81 * 0.5), rel=1e-3)
    assert o["_ledger_rel"] < 0.01


def test_same_answer_as_1d():
    """A 1d spring-mass-dashpot written as 2d (dof [x], linear measure) gives the 1d numbers."""
    one = load_system("""simblocks: 1
name: a
dim: 1d
bodies:
  wall: {ground: true}
  m: {dof: translation, mass: 2 kg, v0: 3 m/s}
elements:
  k: {type: kelvin_voigt, k: 5000 N/m, c: 4 N s/m, measure: m - wall}
outputs:
  xmax: {body: m, quantity: position, reduce: max}
  F: {element: k, quantity: force, reduce: max_abs}
run: {t_end: 0.2 s, dt: 1e-5 s}
""").run()
    S, two = run("""bodies:
  wall: {ground: true}
  m: {dof: [x], mass: 2 kg, v0: [3 m/s, 0]}
elements:
  k: {type: kelvin_voigt, k: 5000 N/m, c: 4 N s/m, measure: m.x}
outputs:
  xmax: {body: m, quantity: position, reduce: max}
  F: {element: k, quantity: force, reduce: max_abs}
run: {t_end: 0.2 s, dt: 1e-5 s}
""")
    for k in ("xmax", "F"):
        assert two[k] == pytest.approx(one[k], rel=1e-9)


def test_circle_collision_momentum_and_restitution():
    """Off-centre hit of two equal discs on a frictionless table: momentum is conserved in x and y, an elastic
    contact returns all the kinetic energy, and the struck disc leaves along the line of centres."""
    S, o = run("""bodies:
  a: {dof: [x, y], mass: 1 kg, at: [0, 0], v0: [2 m/s, 0], circles: {c: {r: 10 mm}}}
  b: {dof: [x, y], mass: 1 kg, at: [30 mm, 10 mm], circles: {c: {r: 10 mm}}}
elements:
  hit: {type: hertz, K: 1e12, n: 1.5, measure: {contact: [a.c, b.c]}}
outputs:
  avx: {body: a, quantity: vx}
  avy: {body: a, quantity: vy}
  bvx: {body: b, quantity: vx}
  bvy: {body: b, quantity: vy}
  Fmax: {element: hit, quantity: force, reduce: max}
run: {t_end: 0.01 s, dt: 2e-7 s}
""")
    assert o["avx"] + o["bvx"] == pytest.approx(2.0, abs=1e-6)
    assert o["avy"] + o["bvy"] == pytest.approx(0.0, abs=1e-6)
    ke = 0.5 * (o["avx"] ** 2 + o["avy"] ** 2 + o["bvx"] ** 2 + o["bvy"] ** 2)
    assert ke == pytest.approx(2.0, rel=1e-3)
    # line of centres at contact: dx = 20 mm * cos(phi), dy = 10 mm  -> sin(phi) = 0.5, phi = 30 deg
    assert math.atan2(o["bvy"], o["bvx"]) == pytest.approx(math.radians(30), abs=2e-3)
    assert o["bvx"] ** 2 + o["bvy"] ** 2 == pytest.approx((2 * math.cos(math.radians(30))) ** 2, rel=2e-3)


def test_rolling_disc_on_incline():
    """A solid disc released on a 30 degree slope with enough friction rolls without slipping: a = 2/3 g sin(30)."""
    S, o = run("""gravity: 9.81 m/s^2
params:
  th: 30 deg
derived:
  ex: = cos($th)
  ey: = -sin($th)
bodies:
  slope: {ground: true, lines: {face: {from: [-1 m, 0.5774 m], to: [1 m, -0.5774 m], depth: 0.2 m}}}
  disc:
    mass: 1 kg
    inertia: 0.00045 kg m^2
    at: [0, "= 0.03/cos($th) + 0.0"]
    circles: {rim: {r: 30 mm}}
elements:
  normal: {type: kelvin_voigt, k: 2e6 N/m, c: 400 N s/m, measure: {contact: [disc.rim, slope.face]}}
  grip: {type: contact_friction, contact: normal, mu: 0.8, v_eps: 0.001 m/s}
outputs:
  vx: {body: disc, quantity: vx}
  vy: {body: disc, quantity: vy}
  w: {body: disc, quantity: omega}
run: {t_end: 0.5 s}
""")
    v = math.hypot(o["vx"], o["vy"])
    a_expect = 2 / 3 * 9.81 * 0.5
    assert v == pytest.approx(a_expect * 0.5, rel=0.02)        # starts touching, rolls for 0.5 s
    assert abs(o["w"]) * 0.03 == pytest.approx(v, rel=0.02)    # rolling: v = w r
    assert o["_ledger_rel"] < 0.02


def test_spinning_tooth_on_edge_gives_true_lever_arm():
    """A tooth on a spinning disc strikes a fixed face. The torque that stops the disc must equal the contact force
    times the true perpendicular arm: here the face is radial, so the arm is the tooth radius."""
    S, o = run("""bodies:
  wall: {ground: true, lines: {face: {from: [0, 0.02 m], to: [0, 0.08 m], depth: 0.02 m}}}
  disc:
    dof: [rz]
    inertia: 1e-3 kg m^2
    at: [0, 0]
    angle0: -10 deg
    omega0: 50 rad/s
    points: {tooth: [0.05 m, 0]}
elements:
  hit: {type: kelvin_voigt, k: 1e6 N/m, c: 0, measure: {contact: [disc.tooth, wall.face]}}
outputs:
  w: {body: disc, quantity: omega}
  amax: {body: disc, quantity: angle, reduce: max}
  F: {element: hit, quantity: force, reduce: max}
run: {t_end: 0.07 s}
""")
    # elastic: the disc bounces back at the same speed
    assert o["w"] == pytest.approx(-50, rel=2e-3)
    # energy: 0.5 I w^2 = 0.5 k d^2 -> F = k d = w sqrt(k I); the face is radial so d = r * overshoot angle
    F = 50 * math.sqrt(1e6 * 1e-3)
    assert o["F"] == pytest.approx(F, rel=0.02)
    assert o["amax"] == pytest.approx(math.pi / 2 + F / 1e6 / 0.05, abs=2e-3)


def test_compound_pendulum_with_pin():
    """A 0.4 m bar pinned at one end: small swing period 2 pi sqrt(2 L / 3 g)."""
    S, o = run("""gravity: 9.81 m/s^2
bodies:
  ground: {ground: true, points: {pivot: [0, 0]}}
  bar:
    mass: 1 kg
    inertia: "= 1*0.4**2/12"
    at: [0, 0]
    com: [0, -0.2 m]
    angle0: 5 deg
    points: {end: [0, 0]}
elements:
  hinge: {type: pin, k: 1e7 N/m, c: 50 N s/m, measure: [ground.pivot, bar.end]}
outputs:
  t1: {body: bar, quantity: omega, reduce: first_time_below, value: -1e-9}
  t2: {body: bar, quantity: angle, reduce: first_time_below, value: 0}
run: {t_end: 1 s}
""")
    T = 2 * math.pi * math.sqrt(2 * 0.4 / (3 * 9.81))
    assert o["t2"] == pytest.approx(T / 4, rel=0.01)


def test_spin_only_body_with_offset_com_parallel_axis():
    """A body that only spins, centre of mass 0.3 m off the pivot and no inertia of its own: gravity at the centre
    of mass and the parallel-axis term make it a simple pendulum of length 0.3 m."""
    S, o = run("""gravity: 9.81 m/s^2
bodies:
  arm: {dof: [rz], mass: 1 kg, inertia: 1e-9 kg m^2, com: [0.3 m, 0], angle0: -85 deg}
elements:
  air: {type: dashpot, c: 0, measure: arm.rz}
outputs:
  t: {body: arm, quantity: angle, reduce: first_time_below, value: -90 deg}
run: {t_end: 1 s}
""")
    assert o["t"] == pytest.approx(0.5 * math.pi * math.sqrt(0.3 / 9.81), rel=5e-3)


def test_projectile_needs_no_elements_and_lands():
    """With gravity a body may fly free; a ground contact catches it. Range of a 45 degree shot: v^2 / g."""
    S, o = run("""gravity: 9.81 m/s^2
bodies:
  floor: {ground: true, lines: {top: {from: [-1 m, 0], to: [10 m, 0], depth: 0.5 m}}}
  ball: {dof: [x, y], mass: 0.1 kg, at: [0, 0.0101 m], v0: ["= 5*cos(pi/4)", "= 5*sin(pi/4)"], circles: {c: {r: 10 mm}}}
elements:
  land: {type: kelvin_voigt, k: 1e5 N/m, c: 20 N s/m, measure: {contact: [ball.c, floor.top]}}
outputs:
  t_land: {element: land, quantity: force, reduce: first_time_above, value: 1 N}
  x_land: {body: ball, quantity: x}
  ymax: {body: ball, quantity: y, reduce: max}
run: {t_end: 0.75 s}
""")
    assert o["ymax"] - 0.0101 == pytest.approx(25 * 0.5 / (2 * 9.81), rel=2e-3)
    assert o["t_land"] == pytest.approx(2 * 5 * math.sin(math.pi / 4) / 9.81, rel=5e-3)


def test_dent_on_contact_keeps_plastic_set_and_ledger():
    S, o = run("""params:
  tab: {block: plastic_indentation_table, R: 3 mm, Y: 1400 MPa, Estar: 110 GPa}
bodies:
  plate: {ground: true, lines: {face: {from: [0, -1 m], to: [0, 1 m], depth: 0.01 m}}}
  hammer: {dof: [x], mass: 0.5 kg, at: [20 mm, 0], v0: [-8 m/s, 0], points: {tip: [-15 mm, 0]}}
elements:
  dent: {type: elastoplastic, table: $tab, measure: {contact: [hammer.tip, plate.face]}}
outputs:
  set: {element: dent, quantity: plastic_set, reduce: max}
  v: {body: hammer, quantity: vx}
run: {t_end: 3 ms}
""")
    assert o["set"] > 0
    assert o["v"] > 0                     # bounced off
    assert o["_ledger_rel"] < 0.02


def test_errors_are_clear():
    with pytest.raises(SpecError, match="coordinate"):
        load_system(HEAD + """bodies:
  g: {ground: true}
  m: {dof: [x], mass: 1 kg}
elements:
  s: {type: spring, k: 1, measure: m - g}
outputs: {x: {body: m, quantity: x}}
run: {t_end: 1 s}
""")
    with pytest.raises(SpecError, match="no point, circle or line called"):
        load_system(HEAD + """bodies:
  g: {ground: true}
  m: {mass: 1 kg, inertia: 1, points: {p: [0, 0]}}
elements:
  s: {type: spring, k: 1, measure: {distance: [g.nothere, m.p]}}
outputs: {x: {body: m, quantity: x}}
run: {t_end: 1 s}
""")
    with pytest.raises(SpecError, match="ambiguous"):
        load_system(HEAD + """bodies:
  g: {ground: true, points: {o: [0, 0]}}
  m: {mass: 1 kg, inertia: 1, points: {p: [1, 0]}}
elements:
  s: {type: spring, k: 1, measure: {distance: [g.o, m.p]}}
outputs: {x: {body: m, quantity: position}}
run: {t_end: 1 s}
""")


def test_vertical_spinner_wedge_example():
    """The bundled 2d example: books close on every contact angle, the backstop stops the fold, a glancing
    tooth (contact angle under 90 deg) never bites. Trends across angles are reported, never asserted."""
    import os
    S = load_system(os.path.join(os.path.dirname(__file__), "..", "examples", "systems2d", "vertical_spinner_wedge.yaml"))
    for hit in ("85 deg", "100 deg", "120 deg"):
        o = S.run({"hit": hit})
        assert o["_ledger_rel"] < 0.01
        assert o["face_rot"] < 31.5                  # fold = 30 deg plus backstop compliance
    assert S.run({"hit": "85 deg"})["F_peak"] == 0.0


def test_editor_measure_forms_compile():
    """Every 2d measure form the editor writes compiles: coordinates, distance, along, contact, pin, slip."""
    S, o = run("""gravity: [0, -9.81]
bodies:
  g: {ground: true, points: {o: [0, 0]}, lines: {floor: {from: [-1, -0.5], to: [1, -0.5], depth: 0.1}}}
  a: {mass: 1 kg, inertia: 0.01, at: [0.2, 0], points: {p: [-0.2, 0]}, circles: {w: {r: 0.05}}}
  b: {dof: [x, y], mass: 1 kg, at: [0.5, 0], points: {q: [0, 0]}}
elements:
  hinge: {type: pin, k: 1e6, c: 100, measure: [g.o, a.p]}
  rope: {type: spring, k: 1e4, measure: {distance: [a.w, b.q]}}
  rail: {type: kelvin_voigt, k: 1e5, c: 50, measure: {along: [g.o, b.q], dir: y}}
  ground_hit: {type: kelvin_voigt, k: 1e5, c: 50, measure: {contact: [a.w, g.floor]}}
  scrape: {type: contact_friction, contact: ground_hit, mu: 0.3}
  air: {type: dashpot, c: 0.1, measure: a.x - b.x + 0.1 * a.rz}
outputs:
  ay: {body: a, quantity: y, reduce: min}
run: {t_end: 0.2 s}
""")
    assert math.isfinite(o["ay"]) and o["_ledger_rel"] < 0.05


def test_particle_strikes_end_of_free_bar():
    """Elastic hit at d = L/2 off the bar's centre of mass: J = 2u / (1/m + 1/M + d^2/I); the bar takes J/M and
    spins at J d / I. Checks the torque coupling of a free (x, y, spin) body."""
    m, M, L, u = 0.2, 1.0, 0.4, 3.0
    I = M * L ** 2 / 12; d = L / 2
    S, o = run(f"""bodies:
  bar: {{mass: {M} kg, inertia: {I}, at: [0, 0], circles: {{end: {{at: [0, {d}], r: 5 mm}}}}}}
  ball: {{dof: [x, y], mass: {m} kg, at: [-30 mm, {d}], v0: [{u}, 0], circles: {{c: {{r: 5 mm}}}}}}
elements:
  hit: {{type: hertz, K: 1e11, n: 1.5, measure: {{contact: [ball.c, bar.end]}}}}
outputs:
  bvx: {{body: ball, quantity: vx}}
  Vx: {{body: bar, quantity: vx}}
  w: {{body: bar, quantity: omega}}
run: {{t_end: 0.012 s, dt: 2e-7 s}}
""")
    J = 2 * u / (1 / m + 1 / M + d * d / I)
    assert o["Vx"] == pytest.approx(J / M, rel=5e-3)
    assert o["w"] == pytest.approx(-J * d / I, rel=5e-3)        # pushed +x above the centre: clockwise
    assert o["bvx"] == pytest.approx(u - J / m, rel=5e-3, abs=1e-3)

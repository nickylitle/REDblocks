"""3D tier: analytic checks of the rigid-body engine (z up)."""
import numpy as np
import math

import pytest

from simblocks.compiler.system import load_system
from simblocks.compiler.loader import SpecError

HEAD = "simblocks: 1\nname: t\ndim: 3d\n"


def run(body, **kw):
    S = load_system(HEAD + body)
    return S, S.run(**kw)


def test_same_answer_as_1d():
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
    S, three = run("""bodies:
  wall: {ground: true}
  m: {dof: [z], mass: 2 kg, v0: [0, 0, 3 m/s]}
elements:
  k: {type: kelvin_voigt, k: 5000 N/m, c: 4 N s/m, measure: m.z}
outputs:
  xmax: {body: m, quantity: position, reduce: max}
  F: {element: k, quantity: force, reduce: max_abs}
run: {t_end: 0.2 s, dt: 1e-5 s}
""")
    for k in ("xmax", "F"):
        assert three[k] == pytest.approx(one[k], rel=1e-9)


def test_gyroscope_precession_rate():
    """Disc spinning fast on a horizontal axle, the axle's end on a ball joint: it precesses about z at
    Omega = m g l / (I_s w_s) instead of falling."""
    m, r, l, ws = 1.0, 0.1, 0.1, 500.0
    Is = 0.5 * m * r * r; It = 0.25 * m * r * r
    S, o = run(f"""gravity: 9.81 m/s^2
bodies:
  base: {{ground: true, points: {{pivot: [0, 0, 0]}}}}
  top:
    mass: {m} kg
    inertia: [{Is}, {It}, {It}]
    at: [{l}, 0, 0]
    omega0: [{ws}, 0, 0]
    points: {{end: [-{l}, 0, 0]}}
elements:
  ball_joint: {{type: pin, k: 1e7 N/m, c: 50 N s/m, measure: [base.pivot, top.end]}}
outputs:
  x: {{body: top, quantity: x}}
  y: {{body: top, quantity: y}}
  zmin: {{body: top, quantity: z, reduce: min}}
run: {{t_end: 2 s}}
""")
    Om = m * 9.81 * l / (Is * ws)
    assert math.atan2(o["y"], o["x"]) == pytest.approx(Om * 2.0, rel=0.01)
    assert o["zmin"] > -2e-3                     # it does not fall: it precesses
    assert o["_ledger_rel"] < 0.01


def test_torque_free_tumbling_keeps_energy_and_flips_on_the_middle_axis():
    """Spin about the intermediate principal axis is unstable (the tennis-racket / Dzhanibekov flip); about the
    largest it is stable. No torque: kinetic energy is kept."""
    I = "[1e-3, 2e-3, 3e-3]"
    S, o = run(f"""bodies:
  g: {{ground: true, points: {{far: [0, 1, 0]}}}}
  nut: {{dof: [rotate], inertia: {I}, omega0: [0.001, 20, 0.001], points: {{tip: [0, 0.1, 0]}}}}
elements:
  probe: {{type: dashpot, c: 0, measure: {{distance: [g.far, nut.tip]}}}}
outputs:
  ke0: {{body: nut, quantity: kinetic_energy}}
  reach: {{element: probe, quantity: measure, reduce: max}}
run: {{t_end: 3 s, dt: 2e-5 s}}
""")
    assert o["ke0"] == pytest.approx(0.5 * 2e-3 * (20 ** 2), rel=2e-3)
    assert o["reach"] > 0.19                     # the body's own y axis turned over: tip went to the far side
    S, o = run(f"""bodies:
  g: {{ground: true, points: {{far: [0, 0, 1]}}}}
  nut: {{dof: [rotate], inertia: {I}, omega0: [0.001, 0.001, 20], points: {{tip: [0, 0, 0.1]}}}}
elements:
  probe: {{type: dashpot, c: 0, measure: {{distance: [g.far, nut.tip]}}}}
outputs:
  reach: {{element: probe, quantity: measure, reduce: max}}
run: {{t_end: 3 s, dt: 2e-5 s}}
""")
    assert o["reach"] < 0.01                     # spin about the largest axis is stable: the tip stays put


def test_custom_spin_axis_tooth_bounce():
    """A disc spinning about the axis [1, 1, 0] (a custom vector) strikes a fixed plane with a tooth. Elastic:
    the spin reverses; peak force w sqrt(k I) does not depend on the tooth radius."""
    S, o = run("""bodies:
  wall:
    ground: true
    planes: {face: {at: [0, 0, 0.05], normal: [1, -1, 0], u: [0, 0, 1], size: [0.04, 0.04], depth: 0.02}}
  disc:
    dof: [spin]
    axis: [1, 1, 0]
    inertia: 1e-3 kg m^2
    angle0: 0
    omega0: 50 rad/s
    points: {tooth: [-0.0353553, 0.0353553, 0]}
elements:
  hit: {type: kelvin_voigt, k: 1e6 N/m, c: 0, measure: {contact: [disc.tooth, wall.face]}}
outputs:
  w: {body: disc, quantity: omega}
  F: {element: hit, quantity: force, reduce: max}
run: {t_end: 0.06 s}
""")
    assert o["w"] == pytest.approx(-50, rel=3e-3)
    assert o["F"] == pytest.approx(50 * math.sqrt(1e6 * 1e-3), rel=0.02)


def test_sphere_rolls_down_slope():
    """Solid sphere on a 30 deg plane with grip: a = 5/7 g sin(30)."""
    th = math.radians(30); r = 0.03; m = 0.2; I = 0.4 * m * r * r
    n = (-math.sin(th), 0, math.cos(th))
    S, o = run(f"""gravity: 9.81 m/s^2
bodies:
  slope: {{ground: true, planes: {{face: {{at: [0, 0, 0], normal: [{n[0]}, 0, {n[2]}], size: [3, 3], depth: 0.2}}}}}}
  ball:
    mass: {m} kg
    inertia: {I}
    at: [{r * n[0]}, 0, {r * n[2]}]
    spheres: {{s: {{r: {r}}}}}
elements:
  normal: {{type: kelvin_voigt, k: 2e5 N/m, c: 60 N s/m, measure: {{contact: [ball.s, slope.face]}}}}
  grip: {{type: contact_friction, contact: normal, mu: 0.8, v_eps: 1 mm/s}}
outputs:
  v: {{body: ball, quantity: speed}}
  w: {{body: ball, quantity: omega}}
run: {{t_end: 0.5 s}}
""")
    a = 5 / 7 * 9.81 * math.sin(th)
    assert o["v"] == pytest.approx(a * 0.5, rel=0.02)
    assert o["w"] * r == pytest.approx(o["v"], rel=0.02)


def test_oblique_sphere_collision_momentum():
    S, o = run("""bodies:
  a: {dof: [x, y, z], mass: 1 kg, at: [0, 0, 0], v0: [2, 0, 0], spheres: {s: {r: 10 mm}}}
  b: {dof: [x, y, z], mass: 2 kg, at: [30 mm, 6 mm, 8 mm], spheres: {s: {r: 10 mm}}}
elements:
  hit: {type: hertz, K: 1e12, n: 1.5, measure: {contact: [a.s, b.s]}}
outputs:
  avx: {body: a, quantity: vx}
  bvx: {body: b, quantity: vx}
  avy: {body: a, quantity: vy}
  bvy: {body: b, quantity: vy}
  avz: {body: a, quantity: vz}
  bvz: {body: b, quantity: vz}
  kea: {body: a, quantity: kinetic_energy}
  keb: {body: b, quantity: kinetic_energy}
run: {t_end: 0.01 s, dt: 2e-7 s}
""")
    assert o["avx"] + 2 * o["bvx"] == pytest.approx(2.0, abs=1e-6)
    assert o["avy"] + 2 * o["bvy"] == pytest.approx(0.0, abs=1e-6)
    assert o["avz"] + 2 * o["bvz"] == pytest.approx(0.0, abs=1e-6)
    assert o["kea"] + o["keb"] == pytest.approx(2.0, rel=1e-3)
    assert o["bvz"] / o["bvy"] == pytest.approx(8 / 6, rel=2e-3)    # pushed along the line of centres


def test_errors_are_clear_3d():
    with pytest.raises(SpecError, match="spin body"):
        load_system(HEAD + """bodies:
  g: {ground: true}
  m: {mass: 1 kg, inertia: 1e-3}
elements:
  s: {type: spring, k: 1, measure: m.spin}
outputs: {x: {body: m, quantity: x}}
run: {t_end: 1 s}
""")
    with pytest.raises(SpecError, match="no point, sphere, plane or cylinder"):
        load_system(HEAD + """bodies:
  g: {ground: true}
  m: {mass: 1 kg, inertia: 1e-3, points: {p: [0, 0, 0]}}
elements:
  s: {type: spring, k: 1, measure: {distance: [g.no, m.p]}}
outputs: {x: {body: m, quantity: x}}
run: {t_end: 1 s}
""")


def test_3d_csv_export_and_verify(tmp_path):
    """CSV export of 3d body motion and contact force; the full verification battery passes on the 3d example."""
    import os, csv
    S = load_system(HEAD + f"""gravity: 9.81 m/s^2
bodies:
  floor: {{ground: true, planes: {{top: {{at: [0, 0, 0], normal: [0, 0, 1], size: [1, 1]}}}}}}
  ball: {{mass: 0.2 kg, inertia: 2e-5, at: [0, 0, 0.1], spheres: {{s: {{r: 0.02}}}}}}
elements:
  land: {{type: kelvin_voigt, k: 2e4 N/m, c: 4 N s/m, measure: {{contact: [ball.s, floor.top]}}}}
outputs:
  zmin: {{body: ball, quantity: z, reduce: min}}
exports:
  - {{name: drop, columns: [{{body: ball, quantity: z, unit: mm}}, {{element: land, quantity: force}}], folder: "{tmp_path.as_posix()}", points: 50}}
run: {{t_end: 0.3 s}}
""")
    out, H, paths = S.run_with_exports(model_path=str(tmp_path / "m.yaml"))
    assert len(paths) == 1 and os.path.exists(paths[0])
    rows = list(csv.reader(open(paths[0])))
    assert rows[0][1].startswith("ball z [mm]") and len(rows) > 40
    S2 = load_system(os.path.join(os.path.dirname(__file__), "..", "examples", "systems3d", "horizontal_spinner_wedge.yaml"))
    rep = S2.verify(mc_n=6)
    bad = [c.name for c in rep.checks if c.status == "FAIL"]
    assert not bad, bad


def test_inertia_cad_form():
    """{Ixx, Iyy, Izz, Pxy, Pxz, Pyz} from CAD: tensor off-diagonals are minus the products; bad signs refused."""
    from simblocks.compiler.model3d import _inertia
    from simblocks.compiler.loader import SpecError
    I = _inertia({"Ixx": 2.0, "Iyy": 2.5, "Izz": 3.0, "Pxy": 0.5, "Pyz": -0.2}, {}, "w")
    assert np.allclose(I, [[2, -0.5, 0], [-0.5, 2.5, 0.2], [0, 0.2, 3]])
    assert np.allclose(_inertia({"Ixx": 2.0, "Iyy": 2.5, "Izz": 3.0, "Ixy": -0.5}, {}, "w"), I + [[0, 0, 0], [0, 0, -0.2], [0, -0.2, 0]])
    with pytest.raises(SpecError):
        _inertia({"Ixx": 1, "Iyy": 1, "Izz": 1, "Pxy": 2}, {}, "w")
    # a box with principal moments (1, 2, 3) turned 30 deg about z: same tumbling energy either way
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Iw = Rz @ np.diag([1e-3, 2e-3, 2.5e-3]) @ Rz.T
    S = load_system(f"""simblocks: 1
name: t
dim: 3d
bodies:
  g: {{ground: true, points: {{o: [0, 0, 0]}}}}
  a: {{dof: free, mass: 1, inertia: {{Ixx: {Iw[0, 0]}, Iyy: {Iw[1, 1]}, Izz: {Iw[2, 2]}, Pxy: {-Iw[0, 1]}}},
      omega0: [{c * 4}, {s * 4}, 0], points: {{p: [0, 0, 0]}}}}
elements:
  s: {{type: spring, k: 0, measure: {{distance: [g.o, a.p]}}}}
outputs:
  ke: {{body: a, quantity: kinetic_energy}}
  w: {{body: a, quantity: omega, reduce: max}}
run: {{t_end: 0.5 s}}
""")
    o = S.run()
    assert o["ke"] == pytest.approx(0.5 * 1e-3 * 16, rel=1e-6)     # spinning about its own principal x axis
    assert o["w"] == pytest.approx(4.0, rel=1e-6)


def test_gravity3_planet_surface_value():
    """3D takes gravity: {planet: ...} as the planet's surface value mu / R^2 along -z."""
    import numpy as np
    from simblocks.compiler.model3d import gravity3
    from simblocks.compiler.model import build_context
    doc = {"dim": "3d", "gravity": {"planet": "moon"}}
    g = gravity3(doc, build_context(doc))
    assert np.allclose(g, [0, 0, -4.9028e12 / 1737400.0 ** 2])
    doc = {"dim": "3d", "gravity": {"planet": "mars", "mu": "4.2828e13 m^3/s^2", "radius": "3389.5 km"}}
    assert abs(gravity3(doc, build_context(doc))[2] + 3.728) < 0.01

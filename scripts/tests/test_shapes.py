"""Body shapes: mass and inertia from dimensions and a material, density or mass."""
import math

import numpy as np
import pytest

from simblocks.compiler.system import load_system
from simblocks.compiler.loader import SpecError
from simblocks.compiler.model import compile_model


def _sys(dim, body, el, extra=""):
    return load_system(f"""simblocks: 1
name: t
dim: {dim}
bodies:
  g: {{ground: true, points: {{o: {'[0, 0]' if dim == '2d' else '[0, 0, 0]'}}}}}
  b: {body}
{extra}elements:
  s: {el}
outputs:
  ke: {{body: b, quantity: kinetic_energy}}
run: {{t_end: 0.01 s}}
""")


def test_2d_disc_from_density():
    R, t, rho = 0.04, 0.006, 7850.0
    S = _sys("2d", f"{{dof: planar, shape: {{kind: disc, R: {R}, t: {t}, density: {rho}}}, at: [0, 0.1], "
                   "omega0: 10 rad/s, points: {p: [0, 0]}}", "{type: spring, k: 0, measure: {distance: [g.o, b.p]}}")
    m = rho * math.pi * R * R * t; I = 0.5 * m * R * R
    o = S.run()
    assert o["ke"] == pytest.approx(0.5 * I * 100, rel=1e-9)
    d = S.nominal.cdoc["bodies"]["b"]
    assert d["mass"] == pytest.approx(m) and d["circles"]["outline"]["r"] == pytest.approx(R)


def test_3d_cylinder_axis_and_material():
    """A cylinder with its axis along x spinning about x: KE = 1/2 (m R^2 / 2) w^2; material sets rho."""
    from simblocks.materials import get
    rho = get("4130_normalized").rho; R, L = 0.03, 0.05
    S = _sys("3d", f"{{dof: free, shape: {{kind: cylinder, R: {R}, L: {L}, material: 4130_normalized, axis: x}}, "
                   "omega0: [20, 0, 0], points: {p: [0, 0, 0]}}", "{type: spring, k: 0, measure: {distance: [g.o, b.p]}}")
    m = rho * math.pi * R * R * L
    assert S.run()["ke"] == pytest.approx(0.5 * 0.5 * m * R * R * 400, rel=1e-9)


def test_3d_sphere_rolls_with_outline():
    """Sphere with an auto 'outline' feature rolling down 30 deg: a = 5/7 g sin."""
    S = load_system("""simblocks: 1
name: t
dim: 3d
gravity: 9.81
bodies:
  ramp: {ground: true, planes: {face: {at: [0, 0, 0], normal: [-0.5, 0, 0.8660254037844386], size: [10, 10]}}}
  ball: {dof: free, shape: {kind: sphere, R: 0.02, mass: 0.1}, at: [-0.01, 0, 0.017320508075688773], points: {c: [0, 0, 0]}}
elements:
  touch: {type: spring, k: 1e6, measure: {contact: [ball.outline, ramp.face]}}
  grip: {type: contact_friction, contact: touch, mu: 0.8, v_eps: 1e-5}
outputs:
  vx: {body: ball, quantity: vx}
run: {t_end: 0.2 s}
""")
    o = S.run()
    a = 5 / 7 * 9.81 * 0.5
    assert o["vx"] == pytest.approx(-a * 0.8660254 * 0.2, rel=0.02)


def test_1d_rotation_body_shape():
    S = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  w: {dof: rotation, shape: {kind: disc, R: 0.05, mass: 2}}
elements:
  s: {type: spring, k: 10, measure: w}
outputs:
  th: {body: w, quantity: position}
run: {t_end: 0.01 s}
""")
    assert S.nominal.cdoc["bodies"]["w"]["inertia"] == pytest.approx(0.5 * 2 * 0.05 ** 2)


def test_shape_conflicts_refused():
    with pytest.raises(SpecError, match="both shape and mass"):
        _sys("2d", "{dof: planar, mass: 1, shape: {kind: disc, R: 0.04, mass: 1}, points: {p: [0, 0]}}",
             "{type: spring, k: 0, measure: {distance: [g.o, b.p]}}")
    with pytest.raises(SpecError, match="exactly one of"):
        _sys("2d", "{dof: planar, shape: {kind: disc, R: 0.04}, points: {p: [0, 0]}}",
             "{type: spring, k: 0, measure: {distance: [g.o, b.p]}}")
    with pytest.raises(SpecError, match="not one of"):
        _sys("2d", "{dof: planar, shape: {kind: sphere, R: 0.04, mass: 1}, points: {p: [0, 0]}}",
             "{type: spring, k: 0, measure: {distance: [g.o, b.p]}}")

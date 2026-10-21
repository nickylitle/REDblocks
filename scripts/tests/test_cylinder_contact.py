"""Cylinder contact (3D): a cylinder is a stack of disc slices, each touching a plane at its deepest rim point
(2026-10-20)."""
import math

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError
from simblocks.compiler.expand_cyl import cyl_offsets

G = 9.81


def test_slice_weights_are_the_trapezoid_rule():
    xs, ws = zip(*cyl_offsets(0.04, 3))
    assert np.allclose(xs, [-0.02, 0.0, 0.02]) and np.allclose(ws, [0.25, 0.5, 0.25])
    xs, ws = zip(*cyl_offsets(0.04, 5))
    assert abs(sum(ws) - 1) < 1e-12 and np.allclose(ws, [0.125, 0.25, 0.25, 0.25, 0.125])
    assert cyl_offsets(0.0, 1) == [(0.0, 1.0)]


SLOPE = """simblocks: 1
name: roll
dim: 3d
gravity: 9.81
bodies:
  floor:
    ground: true
    planes: {top: {at: [0, 0, 0], normal: [%(nx)r, 0, %(nz)r]}}
  roller:
    dof: [x, y, z, rotate]
    mass: 2
    inertia: [0.0009, 0.0009, 0.0009]
    at: [%(x0)r, 0, %(z0)r]
    cylinders: {tread: {at: [0, 0, 0], axis: y, r: 30 mm, width: 40 mm}}
elements:
  tire: {type: kelvin_voigt, measure: {contact: [roller.tread, floor.top]}, k: 2e6, c: 400}
  grip: {type: contact_friction, contact: tire, mu: 1.0, v_eps: 0.001}
outputs:
  vx: {body: roller, quantity: vx}
  Fn: {element: tire, quantity: force}
  pen: {element: tire, quantity: measure}
run: {t_end: 0.4, dt: 4e-6}
"""


def test_solid_cylinder_rolls_down_a_slope_at_two_thirds_g_sin():
    th = math.radians(10); r = 0.03
    doc = SLOPE % dict(nx=-math.sin(th), nz=math.cos(th), x0=-r * math.sin(th), z0=r * math.cos(th))
    o, H = load_system(doc).run(history=True, n_log=2001)
    a = 2.0 / 3.0 * G * math.sin(th)                      # I = m r^2 / 2: rolling without slip
    assert abs(o["vx"] - (-a * math.cos(th) * 0.4)) < 0.004 * a * 0.4
    assert abs(o["Fn"] - 2 * G * math.cos(th)) < 1e-3 * 2 * G
    assert abs(o["pen"] - 2 * G * math.cos(th) / 2e6) < 1e-3 * o["pen"]
    F = np.asarray(H["tire.force"]); parts = sum(np.asarray(H[f"tire__d{j}.force"]) for j in range(3))
    assert np.allclose(F, parts) and abs(F[-1] - o["Fn"]) < 1e-6 * o["Fn"]


FLAT = """simblocks: 1
name: stand
dim: 3d
gravity: 9.81
bodies:
  floor:
    ground: true
    planes: {top: {at: [0, 0, 0], normal: z}}
  wheel:
    dof: %(dof)s
    mass: 1
    inertia: %(I)r
    at: [0, 0, %(z0)r]
    omega0: %(w0)r
    cylinders: {tread: {at: [0, 0, 0], axis: [0, %(ay)r, %(az)r], r: 30 mm, width: 40 mm}}
elements:
  tire: {type: kelvin_voigt, measure: {contact: [wheel.tread, floor.top]}, k: 2e6, c: 300}
  %(fric)s
outputs:
  F: {element: tire, quantity: force}
  w: {body: wheel, quantity: omega}
run: {t_end: %(T)r, dt: 4e-6}
"""


def _flat(**kw):
    d = dict(dof="[z]", I=0.001, w0=0.0, ay=1.0, az=0.0, fric="", T=0.1, z0=0.0299951); d.update(kw)
    o, H = load_system(FLAT % d).run(history=True, n_log=101)
    for j in range(3):                                    # the slices themselves, from the history
        o[f"F{j}"] = float(np.asarray(H[f"tire__d{j}.force"])[-1])
    return o


def test_flat_cylinder_shares_its_weight_a_quarter_half_quarter():
    o = _flat()
    assert abs(o["F"] - G) < 1e-3 * G
    assert abs(o["F0"] - G / 4) < 2e-3 * G and abs(o["F1"] - G / 2) < 2e-3 * G and abs(o["F2"] - G / 4) < 2e-3 * G


def test_tipped_cylinder_stands_on_its_edge():
    a = math.radians(5)                                   # axis tipped: the low end of the tread carries it all
    o = _flat(ay=math.cos(a), az=math.sin(a), z0=0.03 * math.cos(a) + 0.02 * math.sin(a) - 1e-6)
    assert abs(o["F"] - G) < 1e-3 * G
    low = min(o["F0"], o["F2"]); high = max(o["F0"], o["F2"])
    assert abs(high - G) < 1e-3 * G and low == 0.0 and o["F1"] == 0.0


def test_scrub_torque_is_that_of_a_uniform_line():
    # standing on its tread and turning about a vertical axis: friction torque mu N w / 4 (a uniform line contact)
    mu, w, I, w0, T = 0.8, 0.04, 2e-4, 20.0, 0.05
    o = _flat(dof="[z, spin]\n    axis: z", I=I, w0=w0, T=T, fric="grip: {type: contact_friction, contact: tire, mu: %r, v_eps: 1e-4}" % mu)
    alpha = mu * G * w / 4 / I
    assert abs((w0 - o["w"]) - alpha * T) < 0.01 * alpha * T


def test_cylinder_contact_refusals():
    bad = FLAT % dict(dof="[z]", I=0.001, w0=0.0, ay=1.0, az=0.0, fric="", T=0.01, z0=0.0299951)
    with pytest.raises(SpecError, match="cannot be shared"):
        load_system(bad.replace("type: kelvin_voigt", "type: friction").replace("k: 2e6, c: 300", "F_c: 1, v_eps: 0.01")).compile()
    sph = bad.replace("planes: {top: {at: [0, 0, 0], normal: z}}", "spheres: {top: {at: [0, 0, 0], r: 0.01}}")
    with pytest.raises(SpecError, match="plane patches only"):
        load_system(sph).compile()
    with pytest.raises(SpecError):
        load_system(bad.replace("width: 40 mm}", "width: 40 mm, slices: 40}")).compile()

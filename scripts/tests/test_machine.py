"""Gear mesh and bolted joint: closed-form checks (rigid-gear limit, Lewis and Hertz stress, Shigley joint)."""
import math

import pytest

from simblocks.compiler import load_system
from simblocks.compiler.machine import lewis_Y


def gear_model(extra="", t_end="40 ms", torque="2 N m", J1="2e-5", J2="1.8e-4"):
    return load_system(f"""simblocks: 1
name: g
dim: 1d
bodies:
  p: {{dof: rotation, inertia: {J1} kg m^2}}
  w: {{dof: rotation, inertia: {J2} kg m^2}}
elements:
  drive: {{type: force, measure: p, P: {torque}}}
  mesh: {{type: gear, measure: p + w, module: 1 mm, teeth: [12, 36], face_width: 8 mm{extra}}}
outputs:
  wp: {{body: p, quantity: velocity}}
  ww: {{body: w, quantity: velocity}}
  Ft: {{element: mesh, quantity: force, reduce: max_abs}}
  sb: {{element: mesh, quantity: bending_stress, reduce: max}}
  sc: {{element: mesh, quantity: contact_stress, reduce: max}}
run: {{t_end: {t_end}}}
""")


def test_gear_ratio_and_rigid_limit():
    o = gear_model().run()
    r1, r2 = 6e-3, 18e-3
    assert o["ww"] / o["wp"] == pytest.approx(-r1 / r2, rel=2e-3)        # external mesh: opposite ways, 3:1
    # rigid gears: J_eff at the pinion = J1 + J2 (r1/r2)^2, so wp = T t / J_eff
    Jeff = 2e-5 + 1.8e-4 * (r1 / r2) ** 2
    assert o["wp"] == pytest.approx(2.0 * 0.04 / Jeff, rel=3e-3)
    assert abs(o["_ledger_rel"]) < 1e-3


def test_gear_stresses_follow_the_tooth_load():
    o = gear_model().run()
    Y = lewis_Y(12)
    assert o["sb"] == pytest.approx(o["Ft"] / (8e-3 * 1e-3 * Y), rel=1e-6)      # Lewis, the 12-tooth pinion
    phi = math.radians(20); mG = 3.0; E, nu = 206e9, 0.3
    I = math.cos(phi) * math.sin(phi) / 2 * mG / (mG + 1)
    Cp = math.sqrt(1 / (math.pi * 2 * (1 - nu ** 2) / E))
    assert o["sc"] == pytest.approx(Cp * math.sqrt(o["Ft"] / (8e-3 * 12e-3 * I)), rel=1e-6)


def test_backlash_delays_the_drive():
    """The pinion crosses half the 0.2 mm play (0.1 mm of arc) on its own before the teeth touch."""
    s = gear_model(", backlash: 0.2 mm", t_end="2 ms")
    s.doc["outputs"]["t0"] = {"body": "w", "quantity": "velocity", "reduce": "first_time_below", "value": -1e-3}
    from simblocks.compiler.model import compile_model, run_model
    o = run_model(compile_model(s.doc))
    th = 0.1e-3 / 6e-3; alpha = 2.0 / 2e-5
    assert o["t0"] == pytest.approx(math.sqrt(2 * th / alpha), rel=0.03)


def bolt_model(load, extra="", t_end="400 ms"):
    return load_system(f"""simblocks: 1
name: b
dim: 1d
bodies:
  lid: {{dof: translation, mass: 0.05 kg}}
elements:
  pull: {{type: force, measure: lid, pulse: {{kind: ramp, to: {load}, duration: 300 ms}}}}
  hold: {{type: bolt, measure: lid, size: M5, grip: 6 mm, grade: "12.9", preload: 4 kN, zeta: 0.7{extra}}}
outputs:
  Fb: {{element: hold, quantity: bolt_force, reduce: final}}
  Fm: {{element: hold, quantity: clamp_force, reduce: final}}
  sb: {{element: hold, quantity: bolt_stress, reduce: final}}
  set: {{element: hold, quantity: plastic_set, reduce: final}}
  x: {{body: lid, quantity: position, reduce: final}}
run: {{t_end: {t_end}}}
""")


def _joint():
    d, p, L, E = 5e-3, 0.8e-3, 6e-3, 207e9
    At = math.pi / 4 * (d - 0.9382 * p) ** 2
    kb = At * E / L
    km = 0.5774 * math.pi * E * d / (2 * math.log(5 * (0.5774 * L + 0.5 * d) / (0.5774 * L + 2.5 * d)))
    return At, kb, km


def test_bolt_shares_the_load_like_shigley():
    At, kb, km = _joint(); C = kb / (kb + km); Fi = 4000.0
    o = bolt_model("2 kN").run()
    assert o["Fb"] == pytest.approx(Fi + C * 2000, rel=2e-3)
    assert o["Fm"] == pytest.approx(Fi - (1 - C) * 2000, rel=2e-3)
    assert o["sb"] == pytest.approx(o["Fb"] / At, rel=1e-9)
    assert o["set"] == 0.0


def test_bolt_separates_then_carries_it_alone():
    At, kb, km = _joint(); Fi = 4000.0
    P_sep = Fi * (1 + kb / km)
    o = bolt_model(f"{1.2 * P_sep:.6g} N").run()
    assert o["Fm"] == 0.0                                  # the joint has opened
    assert o["Fb"] == pytest.approx(1.2 * P_sep, rel=2e-3)  # the bolt alone holds the whole load


def test_bolt_yields_and_loses_preload():
    At, kb, km = _joint(); Fy = 1100e6 * At
    lt = 6e-3; xb = 0.08 * lt; kh = (1220e6 - 1100e6) * At / xb
    o = bolt_model(f"{1.05 * Fy:.6g} N").run()
    assert o["set"] > 0.0
    assert o["Fb"] == pytest.approx(Fy + kh * o["set"], rel=1e-6)          # on the hardening line


def test_bolt_snaps_past_its_ultimate():
    At, kb, km = _joint()
    s = bolt_model(f"{1.3 * 1220e6 * At:.6g} N", t_end="300 ms")
    s.doc["outputs"]["brk"] = {"element": "hold", "quantity": "broken", "reduce": "max"}
    from simblocks.compiler.model import compile_model, run_model
    o = run_model(compile_model(s.doc))
    assert o["brk"] == 1.0 and o["Fb"] == 0.0


def test_bolt_torque_preload():
    o = load_system("""simblocks: 1
name: b
dim: 1d
bodies:
  lid: {dof: translation, mass: 0.05 kg}
elements:
  hold: {type: bolt, measure: lid, size: 1/4-20, grip: 0.25 in, grade: "grade 8", torque: 10 N m}
outputs:
  Fb: {element: hold, quantity: bolt_force, reduce: final}
run: {t_end: 1 ms}
""").run()
    assert o["Fb"] == pytest.approx(10 / (0.2 * 0.25 * 0.0254), rel=1e-3)     # Fi = T / (K d), held at rest


def _overload(peak):
    s = bolt_model("1 N", t_end="500 ms")
    s.doc["elements"]["pull"]["pulse"] = {"kind": "triangle", "peak": f"{peak:.8g} N", "duration": "300 ms"}
    from simblocks.compiler.model import compile_model, run_model
    return run_model(compile_model(s.doc))


def test_bolt_overload_leaves_less_clamp():
    At, kb, km = _joint(); Fy = 1100e6 * At; Fi = 4000.0
    o = _overload(1.001 * Fy)                   # a whisker past yield: a few um of set
    xp = o["set"]
    assert xp > 0
    # back at rest: bolt and members balance at a smaller preload, Fi' = Fi - km kb xp / (kb + km)
    assert o["Fm"] == pytest.approx(Fi - km * kb * xp / (kb + km), rel=5e-3)
    assert o["Fb"] == pytest.approx(o["Fm"], rel=5e-3)


def test_bolt_overload_can_leave_it_loose():
    At, kb, km = _joint(); Fy = 1100e6 * At
    o = _overload(1.05 * Fy)                    # 0.2 mm of set dwarfs the 8 um preload stretch: no clamp left
    assert o["set"] > 1e-4 and o["Fm"] == 0.0 and abs(o["Fb"]) < 1e-3


def test_gear_in_2d():
    o = load_system("""simblocks: 1
name: g2
dim: 2d
bodies:
  p: {dof: [rz], inertia: 2e-5 kg m^2}
  w: {dof: [rz], inertia: 1.8e-4 kg m^2, at: [24 mm, 0]}
elements:
  drive: {type: force, measure: p.rz, P: 2 N m}
  mesh: {type: gear, measure: p.rz + w.rz, module: 1 mm, teeth: [12, 36], face_width: 8 mm}
outputs:
  wp: {body: p, quantity: omega}
  ww: {body: w, quantity: omega}
run: {t_end: 40 ms}
""").run()
    assert o["ww"] / o["wp"] == pytest.approx(-1 / 3, rel=2e-3)


def test_bolt_min_clamp_tracked():
    s = bolt_model("2 kN")
    s.doc["outputs"]["Fm_min"] = {"element": "hold", "quantity": "clamp_force", "reduce": "min"}
    from simblocks.compiler.model import compile_model, run_model
    At, kb, km = _joint(); C = kb / (kb + km)
    o = run_model(compile_model(s.doc))
    assert o["Fm_min"] == pytest.approx(4000 - (1 - C) * 2000, rel=5e-3)   # every aux slot is tracked

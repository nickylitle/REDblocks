"""System-file compiler: errors are precise, each element law matches an analytic answer, and the three
trial models compiled from YAML reproduce their hand-written sims."""
import math
import os
import sys

import numpy as np
import pytest

from simblocks.compiler import load_system, SpecError
from simblocks.contact.hertz import hertz_impact
from simblocks.actuators.motors import spin_up_time

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = os.path.join(HERE, "..", "examples", "systems")
TRI = os.path.join(HERE, "..", "examples", "trials")


def S(body):
    return load_system("simblocks: 1\nname: t\ndim: 1d\n" + body)


BASE_OUT = "\noutputs:\n  x: {body: a, quantity: position, reduce: max}\nrun: {t_end: 1 s, dt: auto}\n"


# ------------------------------------------------------------------ errors name the line and the fix
@pytest.mark.parametrize("body,needle", [
    ("bodies:\n  a: {dof: translation, mass: 1 kgg}\nelements:\n  s: {type: spring, measure: a, k: 100}" + BASE_OUT,
     "unknown unit 'kgg'"),
    ("bodies:\n  a: {dof: translation, mass: 1 kg}\nelements:\n  s: {type: spring, k: 100}" + BASE_OUT,
     "needs a 'measure'"),
    ("bodies:\n  a: {dof: translation, mass: 1 kg}\n  b: {dof: translation, mass: 1 kg}\nelements:\n"
     "  s: {type: spring, measure: a, k: 100}" + BASE_OUT, "body 'b' is not in any element"),
    ("bodies:\n  a: {dof: rotation, mass: 1 kg}\nelements:\n  s: {type: spring, measure: a, k: 100}" + BASE_OUT,
     "takes 'inertia', not 'mass'"),
    ("bodies:\n  a: {dof: translation, mass: 1 kg}\nelements:\n  s: {type: spring, measure: a, k: 100, c: 3}" + BASE_OUT,
     "'c' is not a field of a spring"),
    ("bodies:\n  a: {dof: translation, mass: 1 kg}\nelements:\n  s: {type: spring, measure: a * a, k: 100}" + BASE_OUT,
     "LINEAR"),
    ("bodies:\n  a: {dof: translation, mass: 1 kg}\nelements:\n  s: {type: spring, measure: a, k: {block: circle, D: 1}}"
     + BASE_OUT, "has no argument 'D'"),
    ("bodies:\n  a: {dof: translation, mass: 1 kg}\nelements:\n  s: {type: spring, measure: a, k: $kk}" + BASE_OUT,
     "undefined parameter '$kk'"),
])
def test_errors_are_specific(body, needle):
    with pytest.raises(SpecError) as ei:
        S(body)
    assert needle in str(ei.value)


def test_error_carries_line_number():
    with pytest.raises(SpecError) as ei:
        S("bodies:\n  a: {dof: translation, mass: 1 kg}\nelements:\n  s: {type: spring, measure: a, k: 5 kN/mmm}" + BASE_OUT)
    assert ":7" in str(ei.value) or ":6" in str(ei.value)


def test_unknown_dim_is_refused_clearly():
    with pytest.raises(SpecError) as ei:
        load_system("simblocks: 1\nname: t\ndim: 4d\nbodies: {}\nelements: {}\noutputs: {}\nrun: {t_end: 1}\n")
    assert "dim must be 1d, 2d or 3d" in str(ei.value)


# ------------------------------------------------------------------ element laws vs closed form
def test_spring_mass_period_and_energy():
    s = S("params: {k: 1000 N/m, m: 0.5 kg}\nbodies:\n  a: {dof: translation, mass: $m, v0: 1 m/s}\n"
          "elements:\n  s: {type: spring, measure: a, k: $k}\n"
          "outputs:\n  xmax: {body: a, quantity: position, reduce: max}\n"
          "  tq: {body: a, quantity: velocity, reduce: first_time_below, value: 0}\nrun: {t_end: 0.2 s, dt: 1 us}\n")
    o = s.run()
    w = math.sqrt(1000 / 0.5)
    assert o["xmax"] == pytest.approx(1 / w, rel=1e-4)
    assert o["tq"] == pytest.approx(math.pi / 2 / w, rel=1e-3)
    assert abs(o["_ledger_rel"]) < 1e-3                       # first order in dt (left-sum work)


def test_hertz_impact_matches_analytic():
    s = S("params: {K: 1e9, m: 0.1 kg, v: 2 m/s}\nbodies:\n  a: {dof: translation, mass: $m, v0: $v}\n"
          "elements:\n  c: {type: hertz, measure: a, K: $K}\n"
          "outputs:\n  F: {element: c, quantity: force, reduce: max}\n  d: {element: c, quantity: measure, reduce: max}\n"
          "run: {t_end: 1 ms, dt: 1 ns}\n")
    o = s.run()
    Estar = 1e9 * 3 / 4; R = 1.0                    # K = 4/3 E* sqrt(R) with R = 1
    an = hertz_impact(0.1, 2.0, Estar, R)
    assert o["F"] == pytest.approx(an["F_max"], rel=1e-3)
    assert o["d"] == pytest.approx(an["delta_max"], rel=1e-3)


def test_friction_stopping_distance():
    s = S("bodies:\n  a: {dof: translation, mass: 2 kg, v0: 3 m/s}\n"
          "elements:\n  f: {type: friction, measure: a, F_c: 4 N, v_eps: 1e-4 m/s}\n"
          "outputs:\n  x: {body: a, quantity: position, reduce: final}\nrun: {t_end: 3 s, dt: 10 us}\n")
    assert s.run()["x"] == pytest.approx(3 ** 2 / (2 * 4 / 2), rel=2e-3)     # v^2 / (2 a)


def test_two_body_reactions_conserve_momentum():
    s = S("bodies:\n  a: {dof: translation, mass: 1 kg, v0: 2 m/s}\n  b: {dof: translation, mass: 3 kg}\n"
          "elements:\n  c: {type: kelvin_voigt, measure: a - b, one_sided: true, k: 1e5, c: 50}\n"
          "outputs:\n  va: {body: a, quantity: velocity}\n  vb: {body: b, quantity: velocity}\nrun: {t_end: 50 ms, dt: 1 us}\n")
    o = s.run()
    assert 1 * o["va"] + 3 * o["vb"] == pytest.approx(2.0, rel=1e-9)


def test_drive_matches_spin_up_time_block():
    s = S("params: {J: 7.6e-4 kg m^2}\nbodies:\n  w: {dof: rotation, inertia: $J}\n"
          "elements:\n  m: {type: drive, measure: w, kv: 1100 rpm/V, R: 35 mohm, I_max: 60 A, gear: 1.5, cells: 5,"
          " R0: 0, capacity: 1e6 Ah}\n"
          "outputs:\n  t: {body: w, quantity: velocity, reduce: first_time_above, value: 11400 rpm}\n"
          "run: {t_end: 3 s, dt: 0.1 ms}\n")
    Kt = 60 / (2 * math.pi * 1100)
    an = spin_up_time(7.6e-4, Kt, 0.035, 21.0, 60.0, 11400 * 2 * math.pi / 60, G=1.5)
    assert s.run()["t"] == pytest.approx(an, rel=2e-3)


# ------------------------------------------------------------------ YAML trials == hand-written trials
@pytest.mark.parametrize("name,keys,tol", [
    ("trial3_cf_plate_drop", ["F_plate_peak_N", "plate_defl_mm", "F_floor_peak_N", "fpf_SF"], 1e-9),
    ("trial1_flywheel_spinup", ["t95_s", "rpm_final", "v_min_per_cell", "mAh_used"], 2e-3),
    ("trial2_shaft_bend", ["shaft_set_mm", "dent_mm", "F_contact_peak_kN", "F_shaft_peak_kN"], 3e-3),
])
def test_yaml_trial_matches_hand_written(name, keys, tol):
    sys.path.insert(0, TRI)
    hand = __import__(name).run({})
    comp = load_system(os.path.join(SYS, name + ".yaml")).run()
    for k in keys:
        assert comp[k] == pytest.approx(hand[k], rel=tol), k


def test_block_library_covers_every_block(tmp_path):
    from simblocks.docs.library import write, entries
    from simblocks.registry import REGISTRY, is_public
    from simblocks.docs.figures import F
    E = entries()
    assert {e["name"] for e in E} == {n for n in REGISTRY if is_public(n)}
    assert {e["name"] for e in entries(include_internal=True)} == set(REGISTRY)
    assert all(e["fig"] is None or e["fig"] in F for e in E)        # None: no drawing says anything useful
    assert all(e["usage"] and e["outputs"] for e in E)
    p = write(str(tmp_path / "lib.html"))
    txt = open(p).read()
    assert "<title>Simblocks Block Library</title>" in txt and txt.count('<symbol id="fig-') == len(F)

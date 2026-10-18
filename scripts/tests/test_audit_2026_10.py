"""Regression tests for the October 2026 system-wide audit (five independent auditors plus an example sweep).
Each test is the reproduction of a confirmed bug, reduced to run fast. IDs match the audit table
(A engines, B compiler, C block physics, D editor, E API and studies, S sweep)."""
import json
import math
import os
import tempfile

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ------------------------------------------------------------------------------------------------ A: engines
@pytest.mark.parametrize("dim", ["1d", "2d"])
def test_A3_history_time_matches_logged_state(dim):
    body = ("a: {dof: translation, mass: 1 kg, v0: 1}" if dim == "1d" else
            "a: {dof: [x], mass: 1 kg, v0: [1, 0], points: {p: [0, 0]}}")
    meas = "a" if dim == "1d" else "a.x"
    q = "position" if dim == "1d" else "x"
    txt = f"""simblocks: 1
name: t
dim: {dim}
bodies:
  {body}
elements:
  f: {{type: force, P: 0, measure: {meas}}}
outputs:
  x_at: {{body: a, quantity: {q}, reduce: at, value: 0.5}}
  tmax: {{body: a, quantity: {q}, reduce: time_of_max}}
run: {{t_end: 1 s, dt: 0.01}}
"""
    o, H = load_system(txt).run(history=True, n_log=100)
    assert np.max(np.abs(H[f"a.{q}"] - H["t"])) < 1e-9          # x = t exactly, row by row
    assert H["t"][0] == 0.0 and abs(H["t"][-1] - 1.0) < 1e-12   # starts at 0 and ends with the final state
    assert abs(o["x_at"] - 0.5) < 1e-9 and abs(o["tmax"] - 1.0) < 1e-9


def test_A3_decimated_history_keeps_the_final_state():
    o, H = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg, v0: 1}
elements:
  f: {type: force, P: 0, measure: a}
outputs:
  x: {body: a, quantity: position}
run: {t_end: 1 s, dt: 0.000625}
""").run(history=True, n_log=800)
    assert abs(H["t"][-1] - 1.0) < 1e-9 and abs(H["a.position"][-1] - 1.0) < 1e-9


def test_A1_auto_step_is_stable_for_overdamped_elements():
    o = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg, x0: 10 mm}
elements:
  s: {type: kelvin_voigt, k: 1e4, c: 1e4, measure: a}
outputs:
  x: {body: a, quantity: position}
  xmin: {body: a, quantity: position, reduce: min}
run: {t_end: 1 s, dt: auto}
""").run()
    assert math.isfinite(o["x"]) and o["xmin"] >= 0.0                # overdamped: decays without crossing zero
    assert o["x"] == pytest.approx(10e-3 * math.exp(-1.0), rel=0.03)  # slow root of m s^2 + c s + k: about -1 /s


def test_A1_pure_dashpot_stops_where_it_should():
    o = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg, v0: 1}
elements:
  d: {type: dashpot, c: 1000, measure: a}
outputs:
  x: {body: a, quantity: position}
  diss: {element: d, quantity: dissipated}
run: {t_end: 50 ms, dt: auto}
""").run()
    assert o["x"] == pytest.approx(1e-3, rel=0.02)                  # v0 m / c
    assert o["diss"] == pytest.approx(0.5, rel=0.02) and abs(o["_ledger_rel"]) < 0.03


def test_A4_friction_never_reverses_the_motion():
    o = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg, v0: 1}
elements:
  f: {type: friction, F_c: 10, v_eps: 1e-4, measure: a}
outputs:
  vmin: {body: a, quantity: velocity, reduce: min}
  diss: {element: f, quantity: dissipated}
run: {t_end: 0.2 s, dt: auto}
""").run()
    assert o["vmin"] >= 0.0 and o["diss"] <= 0.5 + 1e-6 and abs(o["_ledger_rel"]) < 0.01


def test_A5_one_sided_pad_does_not_book_its_stored_energy_as_dissipated():
    k = 1e4; x = 9.81 / k
    o = load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  a: {{dof: translation, mass: 1 kg}}
elements:
  g: {{type: force, P: -9.81, measure: a}}
  pad: {{type: kelvin_voigt, one_sided: true, k: {k}, c: 50, measure: -a}}
outputs:
  diss: {{element: pad, quantity: dissipated}}
run: {{t_end: 2 s, dt: 0.1 ms}}
""").run()
    assert o["diss"] == pytest.approx(9.81 * x - 0.5 * k * x * x, rel=0.05)


def test_A6_planet_gravity_swings_a_pivoted_body():
    base = """simblocks: 1
name: t
dim: 2d
gravity: {G}
bodies:
  rod: {{dof: [rz], mass: 1, inertia: 0.0833, at: [0, {Y}], com: [0, -0.5], angle0: 0.3}}
  g: {{ground: true}}
elements:
  f: {{type: force, P: 0, measure: rod.rz}}
outputs:
  wmax: {{body: rod, quantity: omega, reduce: max_abs}}
run: {{t_end: 1 s, dt: 1 ms}}
"""
    u = load_system(base.format(G="9.81", Y=0)).run()
    p = load_system(base.format(G="{planet: earth, atmosphere: none}", Y=6371000)).run()
    assert p["wmax"] == pytest.approx(u["wmax"], rel=0.02)


# ------------------------------------------------------------------------------------------------ B: compiler
SPRING = """simblocks: 1
name: t
dim: 1d
budget: {limit: 1 kg}
bodies:
  m: {dof: translation, mass: 200 g, x0: 1 mm}
elements:
  s: {type: spring, k: 1000, measure: m, mass: 50 g SEG}
outputs:
  x: {body: m, quantity: position}
run: {t_end: 1 ms}
"""


@pytest.mark.parametrize("seg", ["", ", segments: 4"])
def test_B01_budget_counts_a_segmented_springs_mass(seg):
    assert load_system(SPRING.replace(" SEG", seg)).run()["mass_total"] == pytest.approx(0.25)


@pytest.mark.parametrize("n", [1, 6])
def test_B01_budget_counts_a_beams_mass_once(n):
    txt = f"""simblocks: 1
name: arm
dim: 3d
budget: {{limit: 1 kg}}
params: {{do: 12 mm, di: 10 mm, L: 200 mm}}
derived:
  tube: {{block: tube, do: $do, di: $di}}
bodies:
  hub: {{ground: true, points: {{root: [0, 0, 0]}}}}
  motor: {{dof: free, shape: {{kind: cylinder, R: 14 mm, L: 20 mm, mass: 60 g}}, at: [$L, 0, 0], points: {{mount: [0, 0, 0]}}}}
elements:
  arm: {{type: beam, measure: [hub.root, motor.mount], section: $tube, E: 70 GPa, G: 5 GPa, density: 1550 kg/m^3, segments: {n}, orient: z}}
outputs:
  z: {{body: motor, quantity: z, unit: mm}}
run: {{t_end: 1 ms}}
"""
    m_arm = 1550 * math.pi / 4 * (0.012 ** 2 - 0.010 ** 2) * 0.2
    assert load_system(txt).run()["mass_total"] == pytest.approx(0.06 + m_arm, rel=1e-6)


def test_B02_element_mass_in_2d_is_refused_not_ignored():
    with pytest.raises(SpecError, match="no mass of its own"):
        load_system("""simblocks: 1
name: t
dim: 2d
bodies:
  w: {ground: true, points: {p: [0, 0]}}
  a: {dof: [x], mass: 1 kg, points: {p: [0.1, 0]}}
elements:
  s: {type: spring, k: 1000, mass: 50 g, measure: {distance: [w.p, a.p]}}
outputs:
  x: {body: a, quantity: x}
run: {t_end: 1 ms}
""")


def _seq(tmp, hit, seq):
    open(os.path.join(tmp, "hit.yaml"), "w").write(hit)
    p = os.path.join(tmp, "seq.yaml"); open(p, "w").write(seq)
    return p


HIT = """simblocks: 1
name: hit
dim: 1d
ports:
  inputs:
    v_in: {unit: m/s, default: 2 m/s}
  outputs:
    v_out: {output: v}
bodies:
  a: {dof: translation, mass: 1 kg, v0: $v_in}
elements:
  s: {type: kelvin_voigt, measure: a, one_sided: true, k: 1e5, c: 10}
outputs:
  v: {body: a, quantity: velocity, unit: mm/s}
run: {t_end: 30 ms}
"""


def test_B04_block_outputs_are_si_in_expressions():
    with tempfile.TemporaryDirectory() as d:
        p = _seq(d, HIT, """simblocks: 1
name: seq
params: {v: 3 m/s}
models:
  hit: {file: hit.yaml, inputs: {v_in: $v}}
outputs:
  via_port: {expr: "$hit.v_out", unit: m/s}
  via_output: {expr: "$hit.v", unit: m/s}
""")
        o = load_system(p).run()
    assert o["via_output"] == pytest.approx(o["via_port"], rel=1e-9)
    assert o["hit.v"] == pytest.approx(o["via_port"] * 1000, rel=1e-9)    # the results table: its own unit, mm/s


def test_B03_series_cache_key_is_the_content():
    from simblocks.compiler.chain import Series
    t = np.linspace(0, 1, 11)
    a = Series(t, np.r_[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]); b = Series(t, np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
    assert a.key() != b.key()


@pytest.mark.parametrize("expr", ["= 1 +", "= 1 / 0", "= sqrt(-1)", "= 2 ** 10000", "= abs(1000, 2)"])
def test_B05_bad_expressions_are_spec_errors(expr):
    with pytest.raises(SpecError):
        load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  m: {{dof: translation, mass: 1 kg, x0: 1 mm}}
elements:
  s: {{type: spring, k: "{expr}", measure: m}}
outputs:
  x: {{body: m, quantity: position}}
run: {{t_end: 1 ms}}
""").run()


def test_B06_unknown_output_unit_is_found_at_load():
    with pytest.raises(SpecError, match="outputs.x"):
        load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  m: {dof: translation, mass: 1 kg, x0: 1 mm}
elements:
  s: {type: spring, k: 1000, measure: m}
outputs:
  x: {body: m, quantity: position, unit: furlong}
run: {t_end: 1 ms}
""")


@pytest.mark.parametrize("meas", ["2*(m - 0.05*r)", "(m - 0.05*r)"])
def test_B07_parenthesised_1d_measures_compile(meas):
    load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  m: {{dof: translation, mass: 1 kg, x0: 1 mm}}
  r: {{dof: rotation, inertia: 0.01 kg m^2}}
elements:
  s: {{type: spring, k: 1000, measure: "{meas}"}}
outputs:
  x: {{body: m, quantity: position}}
run: {{t_end: 1 ms}}
""")


@pytest.mark.parametrize("meas", ["m ** 2", "m % 2", "m > 1"])
def test_B08_nonlinear_measure_is_a_spec_error(meas):
    with pytest.raises(SpecError, match="LINEAR"):
        load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  m: {{dof: translation, mass: 1 kg, x0: 1 mm}}
elements:
  s: {{type: spring, k: 1000, measure: "{meas}"}}
outputs:
  x: {{body: m, quantity: position}}
run: {{t_end: 1 ms}}
""")


@pytest.mark.parametrize("bad", ["run:\n", "params: [1, 2]\nrun: {t_end: 1 ms}\n", "run: {t_end: 1 ms}\nports: {inputs: [v]}\n",
                                 "run: {t_end: 1 ms}\nbudget: {limit: 1 kg, items: [a]}\n"])
def test_B09_wrongly_shaped_sections_are_spec_errors(bad):
    with pytest.raises(SpecError):
        load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  m: {dof: translation, mass: 1 kg, x0: 1 mm}
elements:
  s: {type: spring, k: 1000, measure: m}
outputs:
  x: {body: m, quantity: position}
""" + bad)


def test_B14_a_blank_csv_cell_is_an_error_not_a_shifted_column():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "x.csv"), "w").write("t,a,b\n0,,2\n1,,3\n")
        p = os.path.join(d, "m.yaml")
        open(p, "w").write("""simblocks: 1
name: t
dim: 1d
bodies:
  m: {dof: translation, mass: 1 kg}
elements:
  push: {type: force, measure: m, csv: {file: x.csv, against: time, x_col: t, y_col: a}}
outputs:
  x: {body: m, quantity: position}
run: {t_end: 0.5 s}
""")
        with pytest.raises(SpecError, match="needs numbers"):
            load_system(p).run()


def test_B13_optimizer_refuses_a_derived_value():
    from simblocks.compiler.optimize import spec
    s = load_system("""simblocks: 1
name: t
dim: 1d
params: {k0: 1000}
derived: {k: = 2 * $k0}
bodies:
  m: {dof: translation, mass: 1 kg, x0: 1 mm}
elements:
  s: {type: spring, k: $k, measure: m}
outputs:
  x: {body: m, quantity: position, reduce: max_abs}
study: {optimize: {vary: {k: [500, 5000]}, minimize: x}}
run: {t_end: 1 ms}
""")
    with pytest.raises(SpecError, match="derived"):
        spec(s.doc, s)


# ------------------------------------------------------------------------------------------------ C: blocks
def test_C1_epp_section_reaches_mp_not_more():
    from simblocks.materials.curves import Curve
    from simblocks.structures.sections.moment_curvature import moment_curvature_rect as mr
    sy, E, b, h = 300e6, 200e9, 0.02, 0.004
    Mp = sy * b * h * h / 4
    eng = Curve("e", np.array([0, sy / E, 0.3]), np.array([0, sy, sy]), E, "flat engineering stress")
    assert mr(eng, b, h, kappa_max=0.6 / h, measure="eng")["M"].max() == pytest.approx(Mp, rel=0.01)
    e = np.linspace(sy / E, 0.35, 200)                     # flat TRUE stress: finite-strain bending is exactly Mp
    tru = Curve("t", np.r_[0, e], np.r_[0, sy, sy / (1 + e[1:])], E, "flat true stress")
    assert mr(tru, b, h, kappa_max=0.6 / h)["M"].max() == pytest.approx(Mp, rel=0.01)
    small = 0.01 / h                                       # small strain: true and engineering agree
    assert mr(tru, b, h, kappa_max=small)["M"][-1] == pytest.approx(mr(tru, b, h, kappa_max=small, measure="eng")["M"][-1], rel=0.005)


@pytest.mark.parametrize("text,key", [("4130 HRC 45", "4130_45HRC"), ("4130 hardened to 45 Rc", "4130_45HRC"),
                                      ("hot rolled 1018", "1018_HR"), ("4140 prehard, 2 pcs", "4140_QT_30HRC"),
                                      ("titanium grade 2", "Ti_Gr2"), ("a36 typical", "A36_typical"),
                                      ("4140 normalized", "4140_normalized"), ("7075", "7075-T6"), ("pc", "PC")])
def test_C2_material_names_resolve_to_the_right_grade(text, key):
    from simblocks.materials.curves import resolve_material
    assert resolve_material(text) == key


def test_C2_a_hardness_the_catalog_lacks_is_refused():
    from simblocks.materials.curves import resolve_material
    with pytest.raises(KeyError, match="HRC"):
        resolve_material("4140 at 50 HRC")


def test_C3_tip_mass_needs_its_area():
    from simblocks.structures.rotating import bar_spin_root_stress
    with pytest.raises(ValueError):
        bar_spin_root_stress(7850, 1000, 0.1, m_tip=0.05)
    s = bar_spin_root_stress(7850, 1000, 0.1, A=100e-6, m_tip=0.05)
    assert s == pytest.approx(7850 * 1e6 * 0.01 / 2 + 0.05 * 1e6 * 0.1 / 100e-6)


def test_C4_load_point_off_the_plate_is_refused():
    from simblocks.materials import get
    from simblocks.structures.plates.rectangular import rect_ss_point_at
    with pytest.raises(ValueError, match="off the plate"):
        rect_ss_point_at(get("6061-T6"), 0.1, 0.1, 0.002, 0.15, 0.05)


def test_C_belt_follows_shigley_17_9_and_checks_the_groove():
    from simblocks.structures.machine_elements import belt_drive
    r = belt_drive(100.0, 0.05, math.pi, mu=0.3, m_per_L=0.1, rpm=5000)
    assert (r["F1"] + r["F2"]) / 2 - r["Fc"] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="radians"):
        belt_drive(100.0, 0.05, math.pi, groove=38)


# ------------------------------------------------------------------------------------------------ E: API, studies
def test_E8_E9_units_come_from_the_compiled_measure():
    from simblocks.compiler.exports import series_unit
    txt = open(os.path.join(ROOT, "examples", "learn", "1d_3_motor_flywheel.yaml")).read()
    seg = txt.replace("    k: $k_shaft\n", "    k: $k_shaft\n    segments: 3\n    mass: 1e-4 kg m^2\n")
    for t in (txt, seg):
        s = load_system(t)
        assert series_unit(s.nominal, s.doc, "shaft.force") == "N m"
        assert series_unit(s.nominal, s.doc, "shaft.measure") == "rad"


def test_E3_batch_writes_numpy_bools():
    from simblocks.batch import _jsonable
    assert json.loads(json.dumps(_jsonable({"ok": np.float64(1) > 0.5, "n": np.int64(3)}))) == {"ok": True, "n": 3}


def test_E11_export_yaml_writes_big_mappings_one_key_per_line():
    from simblocks.app.api import Api
    lay = {f"p:n{i}": [i * 10, i * 20] for i in range(12)}                # over nine keys: one per line
    doc = {"simblocks": 1, "name": "t", "dim": "1d", "editor": {"layout": lay},
           "bodies": {"bar": {"dof": "translation", "mass": "1 kg"}}}
    y = Api().export_yaml(doc)["text"]
    assert "  layout:\n" in y and "bar: {dof: translation, mass: 1 kg}" in y


def test_E10_goal_seek_target_keeps_its_unit():
    from simblocks.app.api import goal_spec
    s = load_system("""simblocks: 1
name: t
dim: 1d
params: {k: 1000}
bodies:
  m: {dof: translation, mass: 1 kg, v0: 1}
elements:
  s: {type: spring, k: $k, measure: m}
outputs:
  F: {element: s, quantity: force, reduce: max_abs, unit: N}
study: {goal_seek: {vary: k, output: F, target: 5 kN}}
run: {t_end: 0.1 s}
""")
    assert goal_spec(s.doc, s)["target"] == pytest.approx(5000.0)


def test_E5_verify_reports_runs_cut_short_by_the_timeout():
    from simblocks.diagnostics.verify import SimSpec, verify_all
    sp = SimSpec(name="t", run=lambda p, dt: {"x": 1.0, "_timed_out": 1.0, "_ledger_rel": 0.0}, params={}, dt=1e-3,
                 metrics=["x"], mc_n=2, uncertain={"a": (0.0, 1.0)})
    rep = verify_all(sp, skip=("sensitivity",))
    assert any(c[0] == "run_timeout" and c[1] == "FAIL" for c in [(x.name, x.status) for x in rep.checks])


def test_S2_knockout_of_a_contact_that_never_touched_is_skipped():
    from simblocks.diagnostics.verify import verify_all
    s = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg, v0: 1}
elements:
  s: {type: spring, k: 1000, measure: a}
  wall: {type: kelvin_voigt, one_sided: true, k: 1e5, c: 10, measure: a - 1}
outputs:
  x: {body: a, quantity: position, reduce: max}
run: {t_end: 0.2 s}
""")
    rep = verify_all(s.spec(mc_n=2))
    st = {c.name: c.status for c in rep.checks}
    assert st["knockout:no wall"] == "SKIP" and st["knockout:no s"] == "PASS"

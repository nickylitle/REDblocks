"""Features and quality-of-life additions after the October 2026 audit (F = feature, Q = quality of life)."""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_F1_winding_current_of_a_motor_behind_an_esc():
    txt = open(os.path.join(ROOT, "examples", "learn", "1d_4_flywheel_circuit.yaml")).read()
    import yaml
    parts = yaml.safe_load(txt)["circuit"]["parts"]
    motor = next(k for k, v in parts.items() if v.get("type") == "motor")
    txt = txt.replace("outputs:\n", f"outputs:\n  Iw: {{part: {motor}, quantity: winding_current, reduce: max}}\n"
                                    f"  Ib: {{part: {motor}, quantity: current, reduce: max_abs}}\n", 1)
    o = load_system(txt).run()
    assert o["Iw"] > 0 and o["Iw"] >= 0.99 * o["Ib"]          # the winding carries at least the bus current


def test_F1_winding_current_is_a_motors_only():
    txt = open(os.path.join(ROOT, "examples", "learn", "1d_4_flywheel_circuit.yaml")).read()
    txt = txt.replace("outputs:\n", "outputs:\n  bad: {part: pack, quantity: winding_current, reduce: max}\n", 1)
    with pytest.raises(SpecError, match="motor"):
        load_system(txt)


@pytest.mark.parametrize("call", [
    lambda: __import__("simblocks.aero.basic", fromlist=["x"]).cd_sphere(1e6),
    lambda: __import__("simblocks.structures.machine_elements", fromlist=["x"]).set_screw_holding(0.03, 0.02),
    lambda: __import__("simblocks.structures.machine_elements", fromlist=["x"]).notch_sensitivity(2.0, 1e-3, 200e6),
    lambda: __import__("simblocks.structures.fatigue", fromlist=["x"]).marin_size_factor(0.3),
    lambda: __import__("simblocks.structures.machine_elements", fromlist=["x"]).shaft_critical_speed(
        200e9, 0.02, 0.3, masses=[1.0], positions=[0.4]),
    lambda: __import__("simblocks.aero.basic", fromlist=["x"]).isa_troposphere(40000)])
def test_F2_handbook_blocks_refuse_outside_their_range(call):
    with pytest.raises(ValueError):
        call()


def test_F2_isa_covers_the_tropopause_and_lower_stratosphere():
    from simblocks.aero.basic import isa_troposphere
    T, p, rho = isa_troposphere(15000)            # geopotential altitude, as the standard tables list it
    assert T == pytest.approx(216.65) and p == pytest.approx(12045, rel=2e-3) and rho == pytest.approx(0.19367, rel=3e-3)
    assert isa_troposphere(25000)[1] == pytest.approx(2511, rel=3e-3)


def test_F3_plane_strain_root_hinge():
    from simblocks.structures.plates.root_hinge import plate_root_hinge
    a = plate_root_hinge("4130_45HRC", 0.03, 0.0017, 0.03)
    b = plate_root_hinge("4130_45HRC", 0.03, 0.0017, 0.03, plane_strain=True)
    assert b["k"] / a["k"] == pytest.approx(1 / (1 - 0.29 ** 2), rel=0.02)
    assert b["Mp"] / a["Mp"] == pytest.approx(2 / math.sqrt(3), rel=1e-3)


def test_F4_plate_case_large_deflection_matches_the_exact_limit_block():
    from simblocks.materials import get
    from simblocks.structures.plates.cases import plate_case
    from simblocks.structures.plates.large_deflection import circular_plate_large_deflection
    m = get("6061-T6")
    q = plate_case("circular", "clamped", "pressure_large", 0.05, 0.002, material="6061-T6", w0=0.006)["q"]
    assert circular_plate_large_deflection(0.05, 0.002, q, E=m.E, nu=m.nu)["w0"] == pytest.approx(0.006, rel=1e-3)


def test_F5_verify_warns_about_a_contact_surface_without_depth():
    from simblocks.compiler.system import contact_depth_notes
    s = load_system(os.path.join(ROOT, "examples", "systems2d", "vertical_spinner_wedge.yaml"))
    notes = contact_depth_notes(s.doc)
    assert all(st == "WARN" and "depth" in d for _, st, d in notes)


def test_F7_a_port_default_can_use_the_models_params():
    o = load_system("""simblocks: 1
name: t
dim: 1d
params: {v0: 2 m/s}
ports:
  inputs:
    v_in: {unit: m/s, default: = 1.5 * $v0}
bodies:
  a: {dof: translation, mass: 1 kg, v0: $v_in}
elements:
  f: {type: force, P: 0, measure: a}
outputs:
  v: {body: a, quantity: velocity}
run: {t_end: 1 ms}
""").run()
    assert o["v"] == pytest.approx(3.0)


@pytest.mark.parametrize("rev,sign", [("false", -1), ("true", 1)])
def test_F8_a_rack_on_either_side(rev, sign):
    o = load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  p: {{dof: rotation, inertia: 1e-4 kg m^2, v0: 10}}
  r: {{dof: translation, mass: 1 kg}}
elements:
  m: {{type: gear, measure: p + r, module: 1 mm, teeth: [20, rack], face_width: 5 mm, rack_reversed: {rev}}}
outputs:
  x: {{body: r, quantity: position}}
run: {{t_end: 5 ms}}
""").run()
    assert math.copysign(1, o["x"]) == sign


def test_Q3_cli_run_out_writes_outputs_and_history():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-m", "simblocks", "run", os.path.join(ROOT, "examples", "learn",
                            "1d_1_mass_spring_damper.yaml"), "--out", d], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=ROOT), timeout=600)
        assert r.returncode == 0, r.stderr
        assert json.load(open(os.path.join(d, "outputs.json")))
        head = open(os.path.join(d, "history.csv")).readline()
        assert head.startswith("t [s],")


def test_Q7_a_bad_uncertain_entry_is_caught_at_load():
    with pytest.raises(SpecError, match="range"):
        load_system("""simblocks: 1
name: t
dim: 1d
params: {k: 1000}
bodies:
  m: {dof: translation, mass: 1 kg, x0: 1 mm}
elements:
  s: {type: spring, k: $k, measure: m}
outputs:
  x: {body: m, quantity: position}
uncertain: {k: 5}
run: {t_end: 1 ms}
""")


def test_1d_resting_impact_does_not_bounce():
    o = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  a: {dof: translation, mass: 1 kg}
elements:
  g: {type: force, P: -9.81, measure: a}
  floor: {type: impact, e: 0.8, measure: -a}
outputs:
  v: {body: a, quantity: velocity}
run: {t_end: 0.5 s, dt: 1 ms}
""").run(history=True)
    o, H = o
    assert np.max(np.abs(H["a.velocity"][len(H["t"]) // 2:])) < 1e-9     # it settles: no restitution chatter
    assert abs(o["_ledger_rel"]) < 0.01

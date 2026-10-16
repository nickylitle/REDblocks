"""Local batch runner: parallel run, results file round trip, resume after an interrupt."""
import pytest
import json
import os
import textwrap

from simblocks.batch import run_batch, load_results


def _write_job(tmp_path, n=6):
    (tmp_path / "toy.py").write_text(textwrap.dedent('''
        import math
        def batch_record(p, dt):
            # damped oscillator, explicit: final amplitude depends on k, c and dt
            x, v = 1.0, 0.0
            for _ in range(int(0.01 / dt)):
                v += dt * (-p["k"] * x - p["c"] * v); x += dt * v
            return dict(out=dict(x_end=x, k=p["k"], tag=p.get("tag", 0.0)))
    '''))
    (tmp_path / "job.yaml").write_text(textwrap.dedent(f'''
        simblocks_batch: 1
        name: toy
        sim: toy.py
        entry: batch_record
        dt: 1.0e-4
        base: {{c: 1.0}}
        configs: {{a: {{}}, b: {{tag: 1.0}}}}
        samples: {{n: {n}, seed: 3, uniform: {{k: [100, 400]}}}}
        dt_check: {{n: 2, config: a, factors: [2]}}
    '''))
    return str(tmp_path / "job.yaml")


def test_batch_parallel_roundtrip_and_resume(tmp_path):
    job = _write_job(tmp_path)
    out = str(tmp_path / "res.zip")
    # simulate an interrupted run: a partial file holding 3 finished runs plus a cut-off line
    run_batch(job, workers=2, out=out, quiet=True)
    full = load_results(out)
    assert full["meta"]["n_runs"] == 6 * 2 + 2 and len(full["runs"]) == 12 and len(full["dt_check"]) == 2
    keys = {r["key"] for r in full["runs"]}
    assert {"run|a|0", "run|b|5"} <= keys
    assert all(r["record"]["out"]["tag"] == 1.0 for r in full["runs"] if r["key"].startswith("run|b|"))
    with open(out + ".partial.jsonl", "w") as fh:
        for r in full["runs"][:3]:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"key": "run|a|9", "cut')
    os.remove(out)
    run_batch(job, workers=2, out=out, quiet=True)
    again = load_results(out)
    same = {r["key"]: r["record"]["out"]["x_end"] for r in full["runs"]}
    assert {r["key"]: r["record"]["out"]["x_end"] for r in again["runs"]} == same      # deterministic, complete
    assert not os.path.exists(out + ".partial.jsonl")


# ---------------------------------------------------------------- desktop editor backend (simblocks.app.api)
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simblocks.app.api import Api, study_plan, parse_values
from simblocks.compiler import load_system


EX = os.path.join(os.path.dirname(__file__), "..", "examples", "systems")

OSC = """simblocks: 1
name: oscillator
dim: 1d
params: {P: 100 N}
bodies:
  body: {dof: translation, mass: 0.5 kg}
  floor: {ground: true}
elements:
  spring: {type: spring, measure: body - floor, k: 20 kN/m}
  load: {type: force, measure: body, P: $P}
outputs:
  defl_mm: {element: spring, quantity: measure, reduce: max, unit: mm}
run: {t_end: 100 ms}
"""


def test_roundtrip_examples_give_identical_runs():
    a = Api()
    for f in sorted(x for x in os.listdir(EX) if x.endswith(".yaml")):
        txt = open(os.path.join(EX, f)).read()
        back = a.export_yaml(a.import_yaml(txt)["doc"])["text"]
        r1, r2 = a.run(txt), a.run(back)
        assert r1["ok"] and r2["ok"], (f, r2.get("error"))
        assert r1["outputs"] == r2["outputs"], f


def test_parse_values_ranges_and_lists():
    assert parse_values("1, 2, 5") == ["1", "2", "5"]
    v = parse_values("50 N:250 N:5")
    assert len(v) == 5 and abs(v[0] - 50) < 1e-9 and abs(v[-1] - 250) < 1e-9
    assert len(parse_values("0:1:11")) == 11


def test_study_plan_is_sweep_times_mc():
    doc = load_system(OSC + "study:\n  sweep: {P: '10 N:100 N:4'}\n  monte_carlo: {n: 8, ranges: {P: [1 N, 2 N]}}\n").doc
    plan, sweeps, mc = study_plan(doc)
    assert len(plan) == 32 and sweeps == ["P"] and mc == ["P"]


def test_study_runs_in_parallel_and_returns_every_run():
    a = Api()
    txt = OSC + "study:\n  sweep: {P: '50 N:150 N:6'}\n"
    assert a.study_start(txt, 2)["ok"]
    for _ in range(600):
        p = a.study_progress()
        if not p["running"]:
            break
        time.sleep(0.05)
    assert p["done"] == 6 and len(p["results"]) == 6 and all(r["ok"] for r in p["results"])
    d = sorted(r["outputs"]["defl_mm"] for r in p["results"])
    assert abs(d[0] - 2 * 50 / 20) < 0.05 and abs(d[-1] - 2 * 150 / 20) < 0.05      # step load: peak = 2 P / k


def test_undamped_oscillator_ledger_closes():
    """Trapezoidal element work: a sudden constant load on an undamped spring over 3 periods closes to ~1%."""
    o = load_system(OSC).run()
    assert o["_ledger_rel"] < 0.015
    assert abs(o["defl_mm"] - 10.0) < 0.01


# ---------------------------------------------------------------------------------------------- v0.6: exports, folders
SPRING = """simblocks: 1
name: export test
dim: 1d
bodies:
  tip: {dof: translation, mass: 1 kg, v0: 1 m/s}
elements:
  s: {type: spring, measure: tip, k: 10 kN/m}
outputs:
  x: {body: tip, quantity: position, reduce: max, unit: mm}
run: {t_end: 0.1 s}
"""


def test_csv_export_written_with_header_and_rows(tmp_path):
    from simblocks.compiler import load_system
    y = SPRING + f"""exports:
  - name: hist
    columns: [{{element: s, quantity: force, unit: N, label: F}}, {{body: tip, quantity: velocity, unit: mm/s}}]
    folder: {tmp_path.as_posix()}
    naming: "{{model}}_{{name}}"
    time_unit: ms
    points: 50
    header: units
"""
    s = load_system(y)
    out, H, paths = s.run_with_exports(model_path=str(tmp_path / "m.yaml"))
    assert len(paths) == 1 and paths[0].endswith("m_hist.csv")
    lines = open(paths[0]).read().splitlines()
    assert lines[0] == "Time [ms],F [N],tip velocity [mm/s]" and len(lines) == 51
    t1, F1, v1 = map(float, lines[1].split(","))
    assert t1 == 0 and abs(v1 - 1000) < 1e-6          # 1 m/s written in mm/s


def test_export_errors_are_specific():
    from simblocks.compiler import load_system
    from simblocks.compiler.loader import SpecError
    bad = SPRING + "exports:\n  - {name: e, columns: [{element: s, quantity: absorbed_work}]}\n"
    with pytest.raises(SpecError, match="no time history"):
        load_system(bad)
    with pytest.raises(SpecError, match="not defined"):
        load_system(SPRING + "exports:\n  - {name: e, columns: [{body: nope}]}\n")


def test_custom_materials_folder(tmp_path):
    from simblocks.materials import custom, get
    from simblocks.materials.curves import resolve_material
    custom.write_template(tmp_path)
    (tmp_path / "mine.yaml").write_text((tmp_path / "_template.yaml").read_text())
    (tmp_path / "bad.yaml").write_text("materials:\n  z: {E: 1 GPa}\n")
    try:
        r = custom.load_folders([str(tmp_path)])
        assert set(r["added"]) == {"S7_58HRC_mine", "CF_plate_mine"} and any("needs" in e for e in r["errors"])
        assert abs(get("S7_58HRC_mine").sy - 1650e6) < 1 and resolve_material("my s7") == "S7_58HRC_mine"
    finally:
        custom.unload()
    with pytest.raises(KeyError):
        get("S7_58HRC_mine")


def test_custom_blocks_folder(tmp_path):
    from simblocks import registry
    from simblocks.compiler import load_system
    registry.write_block_template(tmp_path)
    (tmp_path / "mine.py").write_text((tmp_path / "_template_blocks.py").read_text())
    (tmp_path / "broken.py").write_text("raise RuntimeError('nope')\n")
    try:
        r = registry.load_block_folders([str(tmp_path)])
        assert r["added"] == ["my_cantilever_tip"] and r["errors"] and "broken.py:1" in r["errors"][0]
        y = SPRING.replace("k: 10 kN/m", "k: $arm.k").replace("bodies:", "derived:\n  arm: {block: my_cantilever_tip, E: 200 GPa, I: 1.2e-9 m^4, L: 80 mm}\nbodies:")
        assert load_system(y).run()["x"] > 0
    finally:
        registry.unload_block_folders()
    assert "my_cantilever_tip" not in registry.REGISTRY


def test_given_section_only_I_is_required():
    import math
    from simblocks.structures.sections.properties import given_section
    from simblocks.structures.beams.static import simply_supported_center
    from simblocks.structures.beams.plastic_spring import beam_plastic_spring
    from simblocks.materials import get
    r = simply_supported_center(get("4140_QT_30HRC"), given_section(1e-9), 0.2)
    assert r["k"] > 0 and math.isnan(r["F_y"])                     # stiffness works, stress needs c
    b, h = 0.01, 0.02
    a = beam_plastic_spring("4140", {"rect": [b, h]}, 0.2)
    g = beam_plastic_spring("4140", {"I": b * h ** 3 / 12, "c": h / 2, "Z": b * h * h / 4}, 0.2)
    assert abs(a["F_collapse"] - g["F_collapse"]) / a["F_collapse"] < 1e-9


def test_app_templates_compile_and_run():
    from simblocks.app.api import Api
    A = Api()
    T = A.templates()
    assert [t["name"] for t in T] == ["Drop test", "Mass on a spring and damper", "Motor, gearbox and flywheel spin-up",
                                      "Bounce and roll (2D)", "Pendulum on a hinge", "Spinning top",
                                      "Drone arm under a thrust punch"]
    for t in T:
        r = A.run(t["text"])
        assert r["ok"] and r["outputs"]["_ledger_rel"] < 0.02, (t["name"], r.get("error"))


def test_timeout_stops_a_long_run_and_says_so():
    import time as _t
    from simblocks.compiler import load_system
    from simblocks.compiler.model import parse_timeout
    assert parse_timeout("auto") == "auto" and parse_timeout("none") is None and parse_timeout("2 min") == 120.0
    y = SPRING.replace("run: {t_end: 0.1 s}", "run: {t_end: 2000 s, timeout: 0.5 s}")
    s = load_system(y); s.run(); t0 = _t.time(); out = s.run(); took = _t.time() - t0
    assert out["_timed_out"] == 1.0 and 0 < out["_t_stopped"] < 2000 and took < 2.0
    assert "_timed_out" not in load_system(SPRING).run()

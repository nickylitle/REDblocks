"""Run-time estimate and live progress (2026-10-21): the step cost comes from what the model has, a per-computer speed
factor is learned and kept, and the engine reports how far a run is while it goes."""
import os
import threading
import time

import pytest

from simblocks.compiler import load_system
from simblocks.compiler import model as m

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def _M(rel):
    path = os.path.join(EX, rel)
    os.environ["SIMBLOCKS_MODEL_DIR"] = os.path.dirname(os.path.abspath(path))
    return load_system(open(path).read()).nominal


def test_step_cost_counts_rigid_joints_and_circuits():
    arm, top = _M("learn/3d_3_robot_arm.yaml"), _M("learn/3d_1_spinning_top.yaml")
    assert m.step_cost_us(arm) > 5 * m.step_cost_us(top)        # six rigid joint rows cost far more than nine contacts
    plain, circ = _M("learn/1d_3_motor_flywheel.yaml"), _M("learn/1d_4_flywheel_circuit.yaml")
    assert m.step_cost_us(circ) > 2 * m.step_cost_us(plain)


def test_speed_factor_is_learned_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_SPEED_FILE", str(tmp_path / "speed.json"))
    monkeypatch.setitem(m.RATE, "speed", 1.0)
    M = _M("learn/1d_4_flywheel_circuit.yaml")           # a few seconds: long enough to learn from
    m.run_model(M, timeout=None)
    learned = m.RATE["speed"]
    assert learned != 1.0 and (tmp_path / "speed.json").exists()
    m.RATE["speed"] = 1.0; m._load_speed()
    assert m.RATE["speed"] == pytest.approx(learned)


def test_engine_reports_progress_while_running():
    M = _M("learn/3d_1_spinning_top.yaml")
    seen = []
    box = {}

    def go():
        box["tid"] = threading.get_ident()
        m.run_model(M, timeout=None)
    th = threading.Thread(target=go); th.start()
    t0 = time.time()
    while th.is_alive() and time.time() - t0 < 60:
        if "tid" in box:
            f = m.run_progress(box["tid"])
            if f is not None:
                seen.append(f)
            if len(seen) > 5 and seen[-1] > 0.05:
                m.cancel_runs(box["tid"])
        time.sleep(0.05)
    th.join(); m.uncancel(box["tid"])
    assert seen and max(seen) > 0.0 and all(0.0 <= f <= 1.0 for f in seen) and seen == sorted(seen)

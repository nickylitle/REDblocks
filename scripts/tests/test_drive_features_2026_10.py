"""Drive features added to redo the Flapjack drive sim (October 2026): reversing ESC with drag brake, gearbox
efficiency, geared rotor inertia in a moving frame, the slip-curve tire, and the MCP watchdog and background jobs."""
import math
import os
import time

import numpy as np
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fly(thr, brake="false", regen="true", eta="", t_end="0.6 s", J="2e-4", gear=1, v0=0):
    return load_system(f"""simblocks: 1
name: t
dim: 1d
bodies:
  fw: {{dof: rotation, inertia: {J} kg m^2, v0: {v0}}}
control:
  thr: "{thr}"
elements:
  bearing: {{type: dashpot, c: 1e-9, measure: fw}}
circuit:
  ground: gnd
  parts:
    pack: {{type: voltage_source, between: [bus, gnd], V: 12 V, R: 20 mohm}}
    mot: {{type: motor, between: [bus, gnd], kv: 1000 rpm/V, R: 0.2 ohm, gear: {gear}, {eta} measure: fw,
          esc: {{throttle: {{control: thr, init: 0, max: 1}}, I_max: 15 A, regen: {regen}, brake: {brake}}}}}
outputs:
  w: {{body: fw, quantity: velocity, reduce: final}}
  I: {{part: mot, quantity: winding_current, reduce: max_abs}}
run: {{t_end: {t_end}}}
""").run()


KT = 60 / (2 * math.pi * 1000)


def test_reverse_throttle_spins_backwards_at_the_current_limit():
    o = _fly("-1")
    assert o["w"] == pytest.approx(-15 * KT / 2e-4 * 0.6, rel=0.01) and o["I"] == pytest.approx(15, rel=1e-3)
    assert abs(o["_ledger_rel"]) < 0.01


def test_drag_brake_decays_with_the_motor_time_constant_and_coast_does_not():
    b = _fly("1 if t < 0.3 else 0", brake="true")
    c = _fly("1 if t < 0.3 else 0", brake="false")
    w0 = c["w"]                                              # coasting keeps the speed it had at 0.3 s
    tau = 2e-4 * 0.2 / KT ** 2
    assert b["w"] == pytest.approx(w0 * math.exp(-0.3 / tau), rel=0.01)
    assert abs(b["_ledger_rel"]) < 0.01


def test_throttle_outside_minus_one_to_one_is_refused():
    with pytest.raises(SpecError, match="-1"):
        load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  fw: {dof: rotation, inertia: 2e-4 kg m^2}
elements:
  b: {type: dashpot, c: 1e-9, measure: fw}
circuit:
  ground: gnd
  parts:
    pack: {type: voltage_source, between: [bus, gnd], V: 12 V}
    mot: {type: motor, between: [bus, gnd], kv: 1000 rpm/V, R: 0.2 ohm, measure: fw, esc: {throttle: -1.5}}
outputs:
  w: {body: fw, quantity: velocity, reduce: final}
run: {t_end: 1 ms}
""")


def test_gearbox_efficiency_forward_and_back_driven():
    from simblocks.actuators.motors import backdrive_efficiency
    a = _fly("1", t_end="0.05 s", J="2e-3", gear=10)
    b = _fly("1", t_end="0.05 s", J="2e-3", gear=10, eta="eta: 0.8,")
    assert b["w"] / a["w"] == pytest.approx(0.8, rel=1e-6)
    a = _fly("0", brake="true", t_end="0.002 s", J="2e-3", gear=10, v0=50)
    b = _fly("0", brake="true", t_end="0.002 s", J="2e-3", gear=10, v0=50, eta="eta: 0.8,")
    assert (50 - b["w"]) / (50 - a["w"]) == pytest.approx(1 / backdrive_efficiency(0.8, 10), rel=0.01)


def test_geared_rotor_in_a_moving_frame_conserves_real_angular_momentum():
    G, Jr, IA, JB = 20.0, 1e-6, 2e-3, 1e-5
    o, H = load_system(f"""simblocks: 1
name: t
dim: 3d
gravity: 0
bodies:
  frame: {{dof: free, mass: 1 kg, inertia: [{IA}, {IA}, {IA}], points: {{ax: [0, 0, 0]}}}}
  out: {{dof: free, mass: 0.1 kg, inertia: [{JB}, {JB}, {JB}], points: {{c: [0, 0, 0]}}}}
elements:
  axle: {{type: hinge, rigid: true, axis: z, measure: [frame.ax, out.c]}}
circuit:
  ground: gnd
  parts:
    pack: {{type: voltage_source, between: [bus, gnd], V: 12 V, R: 10 mohm}}
    mot: {{type: motor, between: [bus, gnd], kv: 1000 rpm/V, R: 0.5 ohm, gear: {G}, J_rotor: {Jr} kg m^2,
          measure: {{twist: [frame, out], axis: z}}, esc: {{throttle: 1, I_max: 5 A}}}}
outputs:
  wa: {{body: frame, quantity: wz, reduce: final}}
  wb: {{body: out, quantity: wz, reduce: final}}
run: {{t_end: 0.2 s, dt: 20 us}}
""").run(history=True)
    # the gearbox keeps the rotor's reaction in the frame: wa / wb = -(JB + G Jr) / (IA - (G - 1) Jr)
    assert o["wa"] / o["wb"] == pytest.approx(-(JB + Jr * G) / (IA + Jr * (1 - G)), rel=1e-6)
    assert abs(o["_ledger_rel"]) < 0.01


def _slide(mu_line, v0=2.0):
    return load_system(f"""simblocks: 1
name: t
dim: 3d
gravity: 9.81
bodies:
  floor: {{ground: true, planes: {{top: {{at: [0, 0, 0], normal: [0, 0, 1], size: [20 m, 20 m]}}}}}}
  puck: {{dof: [x, y, z], mass: 1 kg, at: [0, 0, 0.0001], v0: [{v0}, 0, 0], points: {{p: [0, 0, 0]}}}}
elements:
  n: {{type: kelvin_voigt, k: 1e5, c: 300, measure: {{contact: [puck.p, floor.top]}}}}
  f: {{type: contact_friction, contact: n, {mu_line}}}
outputs:
  v: {{body: puck, quantity: vx, reduce: final}}
run: {{t_end: 0.1 s, dt: 20 us}}
""").run()


def test_tire_curve_slides_at_mu_at_large_slip():
    # a puck has no rolling speed: its slip ratio is 1 (well past the peak), so it slides at mu, like plain friction
    a = _slide("mu: 0.5, v_eps: 0.001")
    b = _slide("mu: 0.5, mu_peak: 0.8, slip_peak: 0.12")
    assert b["v"] == pytest.approx(a["v"], rel=2e-3) and a["v"] == pytest.approx(2.0 - 0.5 * 9.81 * 0.1, rel=0.01)


def test_tire_curve_needs_both_keys_and_is_3d_only():
    with pytest.raises(SpecError, match="slip_peak"):
        _slide("mu: 0.5, mu_peak: 0.8")


def test_mcp_watchdog_and_background_jobs():
    from simblocks import mcp_server as M
    assert M._tmo(None) == "model" and M._tmo("none") is None and M._tmo("5 min") == 300 and M._tmo(0) is None
    p = os.path.join(ROOT, "examples", "learn", "1d_1_mass_spring_damper.yaml")
    j = M.tool_start_job("run", p)
    s = M.tool_job_status(j["job_id"], 50)
    assert s["state"] == "done" and s["result"]["outputs"]["peak_spring_N"] == pytest.approx(4.0)
    o = M.tool_run_model(os.path.join(ROOT, "examples", "learn", "3d_3_robot_arm.yaml"), timeout=1e-3)
    assert o["checks"].get("_timed_out") == 1.0


def test_cache_guard_hash_tracks_sources():
    from simblocks import _cacheguard as g
    assert g.source_hash() == g.source_hash() and len(g.source_hash()) == 40

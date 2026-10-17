"""Control signals: expressions, if / else, PID, lag, rate limit, integrator, fed back into element fields."""
import math

import pytest

from simblocks.compiler.loader import SpecError
from simblocks.compiler.system import load_system

BASE_1D = """
simblocks: 1
name: t
dim: 1d
units: SI
bodies:
  car: {dof: translation, mass: 1200 kg}
  road: {ground: true}
elements:
  air: {type: drag, measure: car - road, K: 0.4}
  roll: {type: dashpot, measure: car - road, c: 15}
  engine: {type: force, measure: car - road, P: {control: push}}
"""


def run(text):
    return load_system(text).run()


def test_pid_cruise_control_settles_on_the_force_balance():
    o = run(BASE_1D + """
params:
  v_set: 25
control:
  push: {pid: {measure: car.velocity, setpoint: $v_set, kp: 800, ki: 120, min: 0, max: 4000}}
outputs:
  v: {body: car, quantity: velocity, reduce: final}
  push_end: {control: push, reduce: final}
  push_max: {control: push, reduce: max}
run: {t_end: 60 s, dt: 1 ms}
""")
    assert o["v"] == pytest.approx(25, rel=1e-4)
    assert o["push_end"] == pytest.approx(0.4 * 25 ** 2 + 15 * 25, rel=1e-3)     # drag + rolling at 25 m/s
    assert o["push_max"] == pytest.approx(4000)                                    # saturated while it speeds up
    assert o["_ledger_rel"] < 1e-4


def test_if_latch_hold_and_the_clock():
    o = run(BASE_1D + """
control:
  push: "= 2000 if t < 10 else 0"
  fast: {if: "car.velocity > 10", then: 1, else: 0, latch: true}
  held: {if: "car.velocity > 10", then: 1, else: 0, for: 2 s}
outputs:
  t_fast: {control: fast, reduce: first_time_above, value: 0.5}
  t_held: {control: held, reduce: first_time_above, value: 0.5}
  fast_end: {control: fast, reduce: final}
  held_end: {control: held, reduce: final}
  t_v10: {body: car, quantity: velocity, reduce: first_time_above, value: 10}
run: {t_end: 120 s, dt: 1 ms}
""")
    assert o["t_fast"] == pytest.approx(o["t_v10"], abs=2e-3)
    assert o["t_held"] == pytest.approx(o["t_v10"] + 2.0, abs=3e-3)
    assert o["fast_end"] == 1.0 and o["held_end"] == 0.0                # latched stays on; held drops when it slows


def test_lag_rate_limit_integrator_against_closed_form():
    o = run(BASE_1D + """
control:
  push: "= 0"
  step: "= 1 if t >= 1 else 0"
  lag1: {lag: step, tau: 0.5 s}
  ramp: {rate_limit: step, rate: 0.25}
  area: {integrate: t}
outputs:
  lag_at: {control: lag1, reduce: at, value: 1.5}
  ramp_at: {control: ramp, reduce: at, value: 3}
  area_end: {control: area, reduce: final}
run: {t_end: 4 s, dt: 0.1 ms}
""")
    assert o["lag_at"] == pytest.approx(1 - math.exp(-1.0), abs=2e-3)
    assert o["ramp_at"] == pytest.approx(0.5, abs=2e-3)
    assert o["area_end"] == pytest.approx(8.0, rel=1e-3)                  # integral of t from 0 to 4


def test_expression_functions():
    o = run(BASE_1D + """
control:
  push: "= 0"
  a: "= clip(7, 0, 5) + max(1, 2) - min(3, 4) + abs(-2) + sqrt(16) + floor(2.7) + sign(-3)"
  b: "= atan2(1, 1) * 4 + (1 if 2 > 1 and not 0 else 5) + (3 >= 3) + (1 == 2)"
outputs:
  a: {control: a, reduce: final}
  b: {control: b, reduce: final}
run: {t_end: 0.01 s, dt: 1 ms}
""")
    assert o["a"] == pytest.approx(5 + 2 - 3 + 2 + 4 + 2 - 1)
    assert o["b"] == pytest.approx(math.pi + 1 + 1 + 0)


def test_hover_hold_in_2d_and_3d():
    for dim, coord in (("2d", "y"), ("3d", "z")):
        at = "[0, 0]" if dim == "2d" else "[0, 0, 0]"
        o = run(f"""
simblocks: 1
name: hover
dim: {dim}
units: SI
gravity: 9.81
bodies:
  drone: {{dof: [{coord}], mass: 1.5 kg, at: {at}}}
elements:
  lift: {{type: force, measure: drone.{coord}, P: {{control: thrust}}}}
control:
  thrust: {{pid: {{measure: drone.{coord}, setpoint: 10, kp: 8, ki: 2, kd: 6, min: 0, max: 40}}}}
outputs:
  h: {{body: drone, quantity: {coord}, reduce: final}}
  thrust: {{control: thrust, reduce: final}}
run: {{t_end: 30 s, dt: 1 ms}}
""")
        assert o["h"] == pytest.approx(10, abs=1e-3), dim
        assert o["thrust"] == pytest.approx(1.5 * 9.81, rel=1e-3), dim


def test_rocket_throttle_switched_by_air_pressure():
    o = run("""
simblocks: 1
name: staged throttle
dim: 2d
units: SI
gravity: {planet: earth}
bodies:
  r: {dof: [x, y], mass: 10000 kg, at: [0, 6371000]}
elements:
  eng: {type: rocket, measure: r, thrust_vac: 250000, mass_flow: 80, throttle: {control: thr, init: 0.7}, stop: 60 s}
  air: {type: aero_drag, measure: r, area: 1, Cd: 0.3}
control:
  thin: {if: "air.ambient_pressure < 50000", then: 1, else: 0, latch: true}
  thr: "= 0.7 + 0.3 * thin"
outputs:
  t_thin: {control: thin, reduce: first_time_above, value: 0.5}
  thr_min: {element: eng, quantity: throttle, reduce: min}
  thr_end: {element: eng, quantity: throttle, reduce: final}
  h_thin: {element: air, quantity: measure, reduce: final}
run: {t_end: 60 s, dt: 10 ms}
""")
    assert 20 < o["t_thin"] < 50                                          # US76: 50 kPa near 5.5 km
    assert o["thr_min"] == pytest.approx(0.7) and o["thr_end"] == pytest.approx(1.0)


def test_driven_spring_keeps_the_ledger():
    o = run("""
simblocks: 1
name: pumped spring
dim: 1d
units: SI
bodies:
  m: {dof: translation, mass: 1 kg, x0: 0.01}
  g: {ground: true}
elements:
  s: {type: spring, measure: m - g, k: {control: kk, init: 1000}}
  d: {type: dashpot, measure: m - g, c: 0.5}
control:
  kk: "= 1000 + 500 * sin(30 * t)"
outputs:
  x: {body: m, quantity: position, reduce: max_abs}
run: {t_end: 5 s, dt: 0.1 ms}
""")
    assert o["x"] > 0.02                       # parametric pumping at twice the natural frequency grows the swing
    assert o["_ledger_rel"] < 1e-3


def test_errors_are_plain():
    with pytest.raises(SpecError, match="no control signal named 'nope'"):
        run(BASE_1D + "control:\n  push: '= 1'\nelements_x: 1\n".replace("elements_x: 1\n", "") +
            "outputs:\n  a: {control: nope}\nrun: {t_end: 0.01 s}\n")
    with pytest.raises(SpecError, match="not a control signal"):
        run(BASE_1D + "control:\n  push: '= missing * 2'\noutputs:\n  a: {control: push}\nrun: {t_end: 0.01 s}\n")
    with pytest.raises(SpecError, match="not a body or an element"):
        run(BASE_1D + "control:\n  push: '= truck.velocity'\noutputs:\n  a: {control: push}\nrun: {t_end: 0.01 s}\n")


STIFF = """
simblocks: 1
name: stiffening spring
dim: 1d
units: SI
bodies:
  m: {dof: translation, mass: 1 kg, x0: 0.01}
  g: {ground: true}
elements:
  s: {type: spring, measure: m - g, k: K_FIELD}
  d: {type: dashpot, measure: m - g, c: 0.5}
control:
  kk: "= 1000 + 20000 * min(t, 1)"
outputs:
  x: {body: m, quantity: position, reduce: max_abs}
run: {t_end: 2 s}
"""


def test_max_sizes_the_time_step_and_a_missing_max_is_flagged():
    from simblocks.compiler.model import control_messages
    no_max = load_system(STIFF.replace("K_FIELD", "{control: kk, init: 1000}"))
    with_max = load_system(STIFF.replace("K_FIELD", "{control: kk, init: 1000, max: 21000}"))
    assert with_max.nominal.dt_crit == pytest.approx(2 / math.sqrt(21000), rel=1e-6)   # sized for the stiffest it gets
    assert with_max.nominal.dt < no_max.nominal.dt
    o1, o2 = no_max.run(), with_max.run()
    assert any("has no max" in m for m in control_messages(o1))
    assert not control_messages(o2)
    assert o2["_ledger_rel"] < 0.01
    over = load_system(STIFF.replace("K_FIELD", "{control: kk, init: 1000, max: 5000}")).run()
    assert any("past its max" in m for m in control_messages(over))


def test_circuit_esc_throttle_on_a_speed_loop():
    import os, yaml
    here = os.path.join(os.path.dirname(__file__), "..", "examples", "learn")
    d = yaml.safe_load(open(os.path.join(here, "1d_4_weapon_circuit.yaml")))
    for p in d["circuit"]["parts"].values():
        if p["type"] == "motor":
            p["esc"]["throttle"] = {"control": "thr", "init": 0}
    d["control"] = {"thr": {"pid": {"measure": "drum.velocity", "setpoint": "8000 rpm", "kp": 0.01, "ki": 0.05,
                                    "min": 0, "max": 1}}}
    d["outputs"] = {"w": {"body": "drum", "quantity": "velocity", "unit": "rpm"}, "thr": {"control": "thr"}}
    d.pop("editor", None)
    o = load_system(yaml.safe_dump(d)).run()
    assert o["w"] == pytest.approx(8000, rel=1e-3)
    assert 0.2 < o["thr"] < 0.9


def test_a_field_stored_inverted_can_be_driven():
    import os, yaml
    here = os.path.join(os.path.dirname(__file__), "..", "examples", "learn")
    base = yaml.safe_load(open(os.path.join(here, "1d_3_motor_flywheel.yaml")))
    base.pop("editor", None)
    en = next(k for k, v in base["elements"].items() if v["type"] == "drive")
    ref = load_system(yaml.safe_dump(base)).run()
    base["elements"][en]["kv"] = {"control": "kvs", "init": base["elements"][en]["kv"]}
    base["control"] = {"kvs": "= " + str(base["elements"][en]["kv"]["init"])}
    o = load_system(yaml.safe_dump(base)).run()
    for k in ref:
        if not k.startswith("_"):
            assert o[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12), k

"""Adaptive time step (run.dt: adaptive): large steps in free motion, the contact step while contacts touch or are
about to, and a step that follows a driven stiffness. Checked against fixed-step runs of the same models (the
reference numbers below are fixed-step results: 1D at 8 us, 2D at the automatic step, the stiffening spring at 2 us)."""
import pytest

from simblocks.compiler.system import load_system

BOUNCE_1D = """simblocks: 1
name: bouncing mass on a stiff stop
dim: 1d
params:
  m: {value: 2 kg}
bodies:
  mass: {dof: translation, mass: $m, x0: 0.5 m}
elements:
  weight:
    type: force
    P: -19.62 N
    measure: mass
  stop:
    type: kelvin_voigt
    one_sided: true
    k: 5e7 N/m
    c: 200 N s/m
    measure: -mass
outputs:
  peak_stop_N: {element: stop, quantity: force, reduce: max_abs, unit: N}
  stop_diss: {element: stop, quantity: dissipated, unit: J}
  top: {body: mass, quantity: position, reduce: max, unit: m}
  v_end: {body: mass, quantity: velocity, unit: m/s}
  x_end: {body: mass, quantity: position, unit: m}
  t_hit: {element: stop, quantity: force, reduce: first_time_above, value: 1 N}
run:
  t_end: 2 s
  dt: adaptive
"""

BOUNCE_2D = """simblocks: 1
name: 2D-2 Ball bouncing down a ramp
question: "A rubber ball dropped onto a 20 degree ramp: how high does it bounce, how much energy is dissipated, and how fast is it rolling at the end?"
dim: 2d
gravity: 9.81

params:
  e: {value: 0.7, note: "coefficient of restitution: bounce speed / impact speed"}
  mu: {value: 0.6, note: "friction coefficient between ball and ramp"}

bodies:
  ground:
    ground: true
    lines:
      ramp: {from: [-1.0, 0.363970], to: [1.0, -0.363970]}
      floor: {from: [1.0, -0.363970], to: [8, -0.363970]}   # long enough that it is still rolling on it at the end
    note: "a 20 degree ramp running into a flat floor"
  ball:
    dof: planar
    shape: {kind: disc, R: 20 mm, t: 40 mm, material: TPU95A_print}   # a disc in 2D; gives it an 'outline' circle
    at: [-0.6, 0.6]
    note: "free in x, y and spin"

elements:
  hit_ramp:
    type: kelvin_voigt
    k: 2e6 N/m
    c: 20 N s/m
    measure: {contact: [ball.outline, ground.ramp]}
    label: ball on the ramp
  hit_floor:
    type: kelvin_voigt
    k: 2e6 N/m
    c: 20 N s/m
    measure: {contact: [ball.outline, ground.floor]}
    label: ball on the floor

outputs:
  dissipated_ramp_J:  {element: hit_ramp, quantity: dissipated, unit: J, label: energy dissipated on the ramp}
  dissipated_floor_J: {element: hit_floor, quantity: dissipated, unit: J, label: energy dissipated on the floor}
  vx_end:       {body: ball, quantity: vx, unit: m/s, label: speed along the floor at the end}
  spin_end:     {body: ball, quantity: omega, unit: rad/s, label: spin at the end}
  ke_end_J:     {body: ball, quantity: kinetic_energy, unit: J}
  peak_ramp_N:  {element: hit_ramp, quantity: force, reduce: max_abs, unit: N}
  x_end:        {body: ball, quantity: x, unit: m}
  y_max:        {body: ball, quantity: y, reduce: max, unit: m}

run:
  t_end: 3 s
  dt: adaptive

"""

STIFFEN = """simblocks: 1
name: a spring that stiffens as it runs
dim: 1d
bodies:
  mass: {dof: translation, mass: 1 kg, v0: 1}
elements:
  spring: {type: spring, measure: mass, k: {control: kk, init: 100}}
  damp: {type: dashpot, measure: mass, c: 0.5}
control:
  kk: "= 100 * 10**(4 * min(t / 2, 1))"
outputs:
  x_end: {body: mass, quantity: position, unit: m}
  v_end: {body: mass, quantity: velocity, unit: m/s}
  vmax: {body: mass, quantity: velocity, reduce: max, unit: m/s}
  diss: {element: damp, quantity: dissipated, unit: J}
run:
  t_end: 3 s
  dt: adaptive
"""


def run(text, **kw):
    return load_system(text).run(**kw)


def close(a, b, rel):
    return abs(a - b) <= rel * abs(b)


def test_1d_drop_on_a_stiff_stop_matches_the_fixed_step():
    o = run(BOUNCE_1D)
    assert o["_steps"] < 30000                      # the fixed step takes 250,000
    assert o["_contact_steps"] > 100 and o["_dt_largest"] > 10 * o["_dt_smallest"]
    assert close(o["peak_stop_N"], 30856.2, 0.002)
    assert close(o["stop_diss"], 1.67901, 0.01)
    assert close(o["t_hit"], 0.31928, 0.001)
    assert close(o["v_end"], -1.66468, 0.01)
    assert close(o["x_end"], 0.273176, 0.02)
    assert o["_ledger_rel"] < 0.005


def test_2d_bounces_down_a_ramp_match_the_fixed_step():
    o = run(BOUNCE_2D)
    assert o["_steps"] < 0.05 * 741860
    assert close(o["peak_ramp_N"], 839.924, 0.003)
    assert close(o["dissipated_ramp_J"], 0.0566706, 0.01)
    assert close(o["dissipated_floor_J"], 0.121725, 0.01)
    assert close(o["vx_end"], 3.14006, 0.005)
    assert close(o["x_end"], 7.22211, 0.005)
    assert o["_ledger_rel"] < 0.01


def test_driven_stiffness_shrinks_the_step():
    s = load_system(STIFFEN); M = s.nominal
    o = s.run()
    # k rises 100 -> 1e6 N/m: the step falls from the end-time cap to 0.02 x 2 / sqrt(1e6) = 40 us
    assert o["_dt_smallest"] == pytest.approx(4e-5, rel=0.05)
    assert o["_dt_largest"] == pytest.approx(1.5e-3, rel=0.01)
    assert close(o["vmax"], 6.04472, 0.002)
    assert close(o["diss"], 12.1372, 0.005)
    assert o["_ledger_rel"] < 0.02
    assert M.dt_fits.shape == (1, 4) and M.dt_fits[0, 1] == pytest.approx(0.5, abs=0.02)


def test_halving_the_step_halves_every_adaptive_step():
    s = load_system(BOUNCE_1D); M = s.nominal
    a = s.run(); b = s.run(dt=M.dt / 2)
    assert b["_dt_largest"] == pytest.approx(a["_dt_largest"] / 2, rel=1e-9)
    assert b["_steps"] > 1.8 * a["_steps"]
    assert abs(b["x_end"] - 0.273176) < abs(a["x_end"] - 0.273176)     # converges toward the fixed-step answer


def test_fixed_step_runs_report_no_adaptive_statistics():
    o = run(BOUNCE_1D.replace("  dt: adaptive\n", "  dt: 20 us\n"))
    assert "_steps" not in o

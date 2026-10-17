"""Motors step exactly: a drive's (or circuit motor's) back-EMF damping on its shaft is integrated over each step as
the exponential it is, so the step is no longer held under twice the motor's mechanical time constant."""
import pytest

from simblocks.compiler.system import load_system

GEARMOTOR = """simblocks: 1
name: stiff gearmotor spins a small arm
dim: 1d
bodies:
  arm: {dof: rotation, inertia: 1e-4 kg m^2}
elements:
  motor:
    type: drive
    measure: arm
    kv: 200 rpm/V
    R: 0.15 ohm
    I_max: 15 A
    gear: 50
    cells: 4
    R0: 0.03 ohm
    capacity: 3000 mAh
  load: {type: dashpot, measure: arm, c: 0.02}
  drag: {type: friction, measure: arm, F_c: 0.3, v_eps: 0.2}
outputs:
  w_end: {body: arm, quantity: velocity, unit: rad/s}
  t_half: {body: arm, quantity: velocity, reduce: first_time_above, value: 6.5}
  i_max: {element: motor, quantity: current, reduce: max}
  w_at: {body: arm, quantity: velocity, reduce: at, value: 0.02}
  i_end: {element: motor, quantity: current}
  soc_end: {element: motor, quantity: soc}
  load_heat: {element: load, quantity: dissipated, unit: J}
run:
  t_end: 0.5 s
"""

CIRCUIT = """simblocks: 1
name: 1D-4 Battery, ESCs and two motors spinning a weapon
question: "How fast do two belt-driven motors on a 6S pack spin a drum up, how hard do they pull on the pack and the fuse, and how hot do the motors get?"
dim: 1d

params:
  kv: {value: 1100 rpm/V, note: "both weapon motors"}
  belt: {value: 2, note: "each motor turns twice per drum turn"}
  I_esc: {value: 80 A, note: "each ESC's current limit"}

bodies:
  drum: {dof: rotation, inertia: 2e-8 kg m^2, label: weapon drum}

elements:
  bearings: {type: friction, measure: drum, F_c: 0.02, v_eps: 1, label: bearing drag (N m)}

circuit:
  ground: gnd
  parts:
    pack: {type: battery, between: [bat, gnd], cells: 6, capacity: 1300 mAh, R0: 18 mohm, R1: 6 mohm, C1: 1500 farad, label: 6S 1300 mAh LiPo}
    fuse: {type: fuse, between: [bat, bus], I_rating: 120 A, t_blow: 2 s, label: main fuse}
    wiring: {type: resistor, between: [bus, esc], R: 4 mohm, label: leads and switch}
    caps: {type: capacitor, between: [esc, gnd], C: 1000 uF, ESR: 20 mohm, label: ESC input caps}
    m1:
      type: motor
      between: [esc, gnd]
      kv: $kv
      R: 32 mohm
      L: 10 uH
      gear: $belt
      measure: drum
      esc: {throttle: 1, I_max: $I_esc}
      thermal: {C_th: 45 J/K, R_th: 3 K/W, T_max_C: 150}
      label: left motor
    m2:
      type: motor
      between: [esc, gnd]
      kv: $kv
      R: 32 mohm
      L: 10 uH
      gear: $belt
      measure: drum
      esc: {throttle: 1, I_max: $I_esc}
      thermal: {C_th: 45 J/K, R_th: 3 K/W, T_max_C: 150}
      label: right motor

outputs:
  t_90_s:       {body: drum, quantity: velocity, reduce: first_time_above, value: 20000 rpm, label: time to 12000 rpm}
  rpm_end:      {body: drum, quantity: velocity, unit: rpm, label: drum speed at the end}
  I_pack_peak:  {part: pack, quantity: current, reduce: min, label: peak pack current (negative = discharging; the first microseconds charge the ESC caps)}
  V_bus_min:    {part: caps, quantity: voltage, reduce: min, label: lowest ESC input voltage}
  fuse_blown:   {part: fuse, quantity: state, reduce: final, label: fuse blown (1) or intact (0)}
  T_motor_max:  {part: m1, quantity: temperature, reduce: max, label: hottest motor temperature (C)}
  soc_end:      {part: pack, quantity: soc, reduce: final, label: pack charge left}
  E_heat_J:     {part: m1, quantity: heat, reduce: final, label: heat in one motor (J)}

run:
  t_end: 0.005 s

"""


def test_a_stiff_gearmotor_runs_at_its_time_constant_not_a_fiftieth_of_it():
    s = load_system(GEARMOTOR); M = s.nominal
    # the old rule: 2 % of the stability limit 2 tau. Now: one tau a step
    assert M.dt == pytest.approx(M.dt_motor, rel=1e-9)
    assert M.dt > 20 * 0.02 * 2 * M.dt_motor


def test_steady_speed_and_current_do_not_depend_on_the_step():
    s = load_system(GEARMOTOR); M = s.nominal
    a = s.run(); b = s.run(dt=M.dt / 25); c = s.run(dt=M.dt * 10)     # c: 10 tau a step, unstable before
    for k in ("w_end", "i_end", "soc_end"):
        assert a[k] == pytest.approx(b[k], rel=1e-5)
        assert c[k] == pytest.approx(b[k], rel=1e-5)
    assert a["load_heat"] == pytest.approx(b["load_heat"], rel=1e-3)
    assert a["_ledger_rel"] < 1e-3


def test_circuit_motor_on_a_light_shaft():
    s = load_system(CIRCUIT); M = s.nominal
    a = s.run(); b = s.run(dt=M.dt / 10)
    assert a["rpm_end"] == pytest.approx(b["rpm_end"], rel=2e-3)
    assert a["E_heat_J"] == pytest.approx(b["E_heat_J"], rel=1e-2)
    assert a["_circuit_ledger_rel"] < 1e-9

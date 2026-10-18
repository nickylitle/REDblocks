"""Small accuracy fixes: event times between steps, battery charge kept 0..100 %, run.accuracy."""
import math

import pytest

from simblocks.compiler.system import load_system

FALL = """simblocks: 1
name: t
dim: 1d
bodies:
  m: {dof: translation, mass: 1 kg}
elements:
  g: {type: force, measure: m, P: -9.81 N}
outputs:
  t_half: {body: m, quantity: position, reduce: first_time_below, value: -0.5}
run: {t_end: 1 s, dt: 10 ms}
"""


def test_event_time_is_interpolated_between_steps():
    o = load_system(FALL).run()
    # the stepped path is x = -g t (t + dt) / 2 (semi-implicit Euler); it crosses -0.5 m at 0.3143 s, between steps
    dt = 0.01; disc = (-dt + math.sqrt(dt * dt + 4 * 0.5 * 2 / 9.81)) / 2
    assert abs(o["t_half"] - disc) < 3e-4                  # without interpolation it reads the step after: 0.32 s


def test_battery_charge_stays_at_full():
    o = load_system("""simblocks: 1
name: t
dim: 1d
bodies:
  m: {dof: translation, mass: 1 kg}
elements:
  s: {type: spring, measure: m, k: 1}
circuit:
  ground: gnd
  parts:
    pack: {type: battery, between: [bus, gnd], cells: 4, capacity: 1000 mAh, R0: 20 mohm}
    charger: {type: current_source, between: [bus, gnd], I: 10 A}
outputs:
  soc: {part: pack, quantity: soc}
run: {t_end: 36 s}
""").run()
    assert o["soc"] == pytest.approx(1.0)


def test_accuracy_scales_the_automatic_step():
    base = load_system(FALL.replace("dt: 10 ms", "dt: auto")).nominal.dt
    fine = load_system(FALL.replace("dt: 10 ms", "dt: auto, accuracy: fine")).nominal.dt
    draft = load_system(FALL.replace("dt: 10 ms", "dt: auto, accuracy: draft")).nominal.dt
    assert fine == pytest.approx(base / 2) and draft == pytest.approx(base * 2)


DENT = """simblocks: 1
name: dent
dim: 1d
bodies:
  m: {dof: translation, mass: 2 kg, v0: -10}
elements:
  stop:
    type: elastoplastic
    measure: -m
    one_sided: true
    table: {k: 1.0e7, acc_tab: [0, 0.05], F_tab: [2000, 2000]}
    RATE
outputs:
  set_mm: {element: stop, quantity: plastic_set, reduce: max, unit: mm}
  Fmax: {element: stop, quantity: force, reduce: max}
  fac: {element: stop, quantity: rate_factor, reduce: max}
run: {t_end: 30 ms}
"""


def test_strain_rate_hardening_raises_the_dynamic_yield():
    a = load_system(DENT.replace("RATE", "")).run()
    b = load_system(DENT.replace("RATE", "strain_rate: {D: 40.4, q: 5, length: 10 mm}")).run()
    c = load_system(DENT.replace("RATE", "strain_rate: {material: 1018_CD, length: 10 mm}")).run()
    assert a["Fmax"] == pytest.approx(2000, rel=1e-6)
    # at the hit 10 m/s over 10 mm is 1000 /s: Cowper-Symonds mild steel 1 + (1000 / 40.4)^(1/5) = 2.90
    assert b["fac"] == pytest.approx(1 + (1000 / 40.4) ** 0.2, rel=1e-3)
    assert b["set_mm"] < 0.5 * a["set_mm"] and b["_ledger_rel"] < 1e-3
    assert c["Fmax"] == pytest.approx(b["Fmax"], rel=1e-9)            # the material's constants are the same

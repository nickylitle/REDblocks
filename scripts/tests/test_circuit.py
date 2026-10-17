"""Electrical circuit parts against their textbook results, and the circuit coupled to the mechanics."""
import math

import numpy as np
import pytest

from simblocks.compiler.circuit import circuit_step, NPRM, NST, NOUT
from simblocks.compiler.loader import SpecError
from simblocks.compiler.system import load_system


def _run(parts, nets, T, dt, speed=0.0, th=None):
    n = len(parts); nn = len(nets) - 1
    pt = np.zeros(n, np.int64); na = np.zeros(n, np.int64); nb = np.zeros(n, np.int64)
    prm = np.zeros((n, NPRM)); pst = np.zeros((n, NST)); pst[:, 4] = 298.15
    for k, (t, a, b, P, S) in enumerate(parts):
        pt[k] = t; na[k] = nets[a]; nb[k] = nets[b]; prm[k, :len(P)] = P; pst[k, :len(S)] = S
    th = np.zeros((n, 8)) if th is None else th
    for k in range(n):
        if th[k, 0] > 0.5:
            pst[k, 4] = th[k, 3]
    pout = np.zeros((n, NOUT)); st = np.zeros((2, 5)); st[0, 2] = speed
    flg = np.zeros(n); G = np.zeros((nn, nn)); I = np.zeros(nn); vn = np.zeros(nn + 1)
    for i in range(int(round(T / dt))):
        circuit_step(i * dt, dt, nn, pt, na, nb, prm, pst, th, pout, np.zeros(1, np.int64), st, flg, G, I, vn)
    return pout, pst, st


N3 = {"gnd": 0, "a": 1, "b": 2}
N2 = {"gnd": 0, "a": 1}


def test_rc_charging_and_rl_rise():
    po, ps, _ = _run([(4, "a", "gnd", [10, 1e-4], []), (1, "a", "b", [1000], []), (2, "b", "gnd", [1e-3], [])], N3, 1.0, 1e-4)
    assert ps[2, 0] == pytest.approx(10 * (1 - math.exp(-1)), rel=1e-3)
    po, ps, _ = _run([(4, "a", "gnd", [10, 1e-4], []), (1, "a", "b", [1.0], []), (3, "b", "gnd", [1.0, 0], [])], N3, 1.0, 1e-4)
    assert ps[2, 0] == pytest.approx(10 * (1 - math.exp(-1)), rel=1e-3)
    assert abs(po[:, 3].sum()) < 1e-6 * np.abs(po[:, 3]).sum()           # Tellegen: the books close


def test_capacitor_esr_limits_the_first_current():
    po, _, _ = _run([(4, "a", "gnd", [10, 1e-6], []), (1, "a", "b", [1.0], []), (2, "b", "gnd", [1e-3, 0, 1.0], [])], N3, 1e-5, 1e-5)
    assert po[2, 0] == pytest.approx(5.0, rel=0.01)


def test_battery_sag_and_charge_count():
    po, ps, _ = _run([(6, "a", "gnd", [6, 3600, 0.1, 0, 0, 4.2, 3.3], [1.0]), (1, "a", "gnd", [1.0], [])], N2, 1e-3, 1e-4)
    assert po[0, 1] == pytest.approx(6 * 4.2 / 1.1, rel=1e-4)
    po, ps, _ = _run([(6, "a", "gnd", [6, 3600, 0.1, 0, 0, 4.2, 3.3], [1.0]), (1, "a", "gnd", [1.0], [])], N2, 10.0, 1e-3)
    i_avg = 0.5 * (6 * 4.2 / 1.1 + po[1, 0])
    assert 1 - ps[0, 0] == pytest.approx(10 * i_avg / 3600, rel=0.01)       # charge drawn = I t / capacity


def test_motor_stall_back_emf_and_esc_limit():
    p = [0.01, 0.1, 0, 1, 0, 0, 0, 0]
    po, _, st = _run([(4, "a", "gnd", [12, 1e-6], []), (10, "a", "gnd", p, [])], N2, 1e-3, 1e-4, speed=0.0)
    assert po[1, 0] == pytest.approx(120, rel=1e-4) and st[0, 0] == pytest.approx(1.2, rel=1e-4)
    po, _, _ = _run([(4, "a", "gnd", [12, 1e-6], []), (10, "a", "gnd", p, [])], N2, 1e-3, 1e-4, speed=600.0)
    assert po[1, 0] == pytest.approx(60, rel=1e-4)
    pe = [0.01, 0.1, 0, 1, 1.0, 20, 0, 0]
    po, _, st = _run([(4, "a", "gnd", [12, 1e-6], []), (11, "a", "gnd", pe, [])], N2, 1e-3, 1e-4)
    assert st[0, 1] == pytest.approx(20, rel=1e-6) and po[1, 0] == pytest.approx(20 * 20 * 0.1 / 12, rel=1e-4)


def test_diode_fuse_and_thermal():
    po, _, _ = _run([(4, "a", "gnd", [5, 1e-6], []), (7, "a", "b", [0.7, 0.01, 1e7], []), (1, "b", "gnd", [100], [])], N3, 1e-3, 1e-4)
    assert po[2, 0] == pytest.approx(4.3 / 100.01, rel=1e-6)
    po, ps, _ = _run([(4, "a", "gnd", [20, 1e-6], []), (9, "a", "b", [1e-3, 10, 0.1, 1e300], []), (1, "b", "gnd", [1.0], [])], N3, 0.03, 1e-4)
    assert ps[1, 5] == 1.0 and abs(po[1, 0]) < 1e-6                     # 400 A^2 s after 25 ms: blown
    th = np.zeros((2, 8)); th[1] = [1, 1.0, 5.0, 298.15, 0.0, 1e9, 0, 0]
    po, _, _ = _run([(4, "a", "gnd", [10, 1e-6], []), (1, "a", "gnd", [10.0], [])], N2, 60, 1e-3, th=th)
    assert po[1, 6] == pytest.approx(75.0, abs=0.05)                     # 10 W x 5 K/W over 25 C


BASE = """simblocks: 1
name: t
dim: 1d
bodies:
  blade: {dof: rotation, inertia: 7.6e-4 kg m^2}
"""


def test_circuit_motor_matches_the_drive_element():
    d = load_system(BASE + """elements:
  mot: {type: drive, kv: 1350 rpm/V, R: 45 mohm, I_max: 60 A, cells: 6, R0: 25 mohm, capacity: 650 mAh, measure: blade}
outputs:
  w: {body: blade, quantity: velocity}
  soc: {element: mot, quantity: soc}
run: {t_end: 2 s}
""").run()
    c = load_system(BASE + """elements: {}
circuit:
  parts:
    pack: {type: battery, between: [bus, gnd], cells: 6, capacity: 650 mAh, R0: 25 mohm}
    wpn: {type: motor, between: [bus, gnd], kv: 1350 rpm/V, R: 45 mohm, measure: blade, esc: {I_max: 60 A}}
outputs:
  w: {body: blade, quantity: velocity}
  soc: {part: pack, quantity: soc}
  E_motor: {part: wpn, quantity: energy}
  heat: {part: wpn, quantity: heat}
run: {t_end: 2 s}
""").run()
    assert c["w"] == pytest.approx(d["w"], rel=2e-3)
    assert c["soc"] == pytest.approx(d["soc"], rel=1e-3)
    KE = 0.5 * 7.6e-4 * c["w"] ** 2
    assert c["E_motor"] == pytest.approx(KE + c["heat"], rel=5e-3)     # electrical in = kinetic + winding heat
    assert c["_circuit_ledger_rel"] < 1e-5


def test_dc_motor_on_a_fixed_supply():
    o = load_system(BASE + """elements:
  mot: {type: dc_motor, kv: 1000 rpm/V, R: 0.1 ohm, V: 12 V, measure: blade}
outputs:
  w: {body: blade, quantity: velocity, unit: rpm}
  I: {element: mot, quantity: current, reduce: max}
run: {t_end: 12 s}
""").run()
    assert o["w"] == pytest.approx(12000, rel=1e-4) and o["I"] == pytest.approx(120, rel=1e-6)


def test_circuit_errors():
    with pytest.raises(SpecError, match="do not connect to the ground"):
        load_system(BASE + """elements: {}
circuit:
  parts:
    r: {type: resistor, between: [a, b], R: 1 ohm}
    m: {type: motor, between: [gnd, c], Kt: 0.01, R: 0.1, measure: blade}
outputs:
  i: {part: r, quantity: current}
run: {t_end: 1 ms}
""").run()
    with pytest.raises(SpecError, match="kv .* or Kt"):
        load_system(BASE + """elements: {}
circuit:
  parts:
    m: {type: motor, between: [a, gnd], R: 0.1, measure: blade}
    v: {type: voltage_source, between: [a, gnd], V: 12 V}
outputs:
  i: {part: m, quantity: current}
run: {t_end: 1 ms}
""").run()


def test_circuit_only_model_needs_no_elements_and_exports_part_columns(tmp_path):
    s = load_system("""simblocks: 1
name: rc export
dim: 1d
circuit:
  parts:
    src: {type: voltage_source, between: [a, gnd], V: 10 V, R: 1 ohm}
    c: {type: capacitor, between: [a, gnd], C: 0.1 farad}
outputs:
  v: {part: c, quantity: voltage, reduce: final}
exports:
  - {name: rc, columns: [{part: c, quantity: voltage}, {part: src, quantity: current}], folder: %s}
run: {t_end: 0.3 s}
""" % tmp_path)
    r, _, paths = s.run_with_exports(model_path=str(tmp_path / "m.yaml"))
    assert r["v"] == pytest.approx(10 * (1 - math.exp(-3)), rel=2e-2)
    rows = open(paths[0]).read().splitlines()
    assert rows[0].startswith("Time [s],c voltage [V],src current [A]")
    assert float(rows[-1].split(",")[1]) == pytest.approx(r["v"], rel=1e-3)

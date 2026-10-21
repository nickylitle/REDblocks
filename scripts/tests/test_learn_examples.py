"""The teaching examples run, close their energy ledger and match their textbook checks."""
import math
import os

import pytest

from simblocks.compiler.system import load_system

HERE = os.path.join(os.path.dirname(__file__), "..", "examples", "learn")


def run(name):
    return load_system(os.path.join(HERE, name)).run()


def test_1d_1_mass_spring_damper():
    o = run("1d_1_mass_spring_damper.yaml")
    zeta, wn = 0.1, 20.0
    Td = 2 * math.pi / (wn * math.sqrt(1 - zeta ** 2))
    assert o["first_trough_mm"] == pytest.approx(-10 * math.exp(-zeta * wn * Td / 2), rel=0.01)
    assert o["dissipated_J"] == pytest.approx(0.5 * 400 * 0.01 ** 2, rel=0.01)     # nearly all the spring energy
    assert o["_ledger_rel"] < 1e-3


def test_1d_2_shock_isolation():
    o = run("1d_2_shock_isolation.yaml")
    assert o["peak_shock_kN"] == pytest.approx(2.0, rel=0.01)
    assert o["peak_speed"] == pytest.approx(2000 * 0.005 * 2 / math.pi / 20, rel=0.03)   # impulse / m
    assert 7 < o["peak_travel_mm"] < 10.5 and 150 < o["peak_floor_N"] < 220
    assert o["_ledger_rel"] < 1e-3


def test_1d_3_motor_flywheel():
    o = run("1d_3_motor_flywheel.yaml")
    assert o["rpm_end"] == pytest.approx(1000 * 16.8 / 5, rel=0.02)
    assert o["peak_current_A"] == pytest.approx(60.0)
    assert 2.5 < o["t_3000rpm_s"] < 3.5
    assert o["_ledger_rel"] < 1e-3


def test_1d_4_flywheel_circuit():
    o = run("1d_4_flywheel_circuit.yaml")
    kv = 1100 * 2 * math.pi / 60 / 1.0                          # rad/s per V
    top = kv * 25.2 / 2 * 60 / (2 * math.pi)                     # no-load flywheel rpm on a full pack
    assert 0.93 * top < o["rpm_end"] < top                       # sag and drag take a little off
    tau = 2 * 2 * (1 / kv) * 80                                  # both ESCs at their limit, through the 2:1 belt
    assert o["t_90_s"] > 12000 * 2 * math.pi / 60 / (tau / 4e-3)  # never faster than the current-limited rate
    assert o["fuse_blown"] == 0 and o["_circuit_ledger_rel"] < 1e-6
    assert o["I_pack_peak"] < -400                               # inrush into the empty ESC caps


def test_2d_1_pendulum():
    o = run("2d_1_pendulum.yaml")
    T0 = 2 * math.pi * math.sqrt(2 * 0.5 / (3 * 9.81))
    assert o["quarter_period_s"] == pytest.approx(T0 / 4 * (1 + math.radians(30) ** 2 / 16), rel=0.005)
    assert o["hinge_drift"] < 1e-5 and o["_ledger_rel"] < 3e-3


def test_2d_2_ramp_bounce():
    o = run("2d_2_ramp_bounce.yaml")
    assert o["vx_end"] == pytest.approx(-o["spin_end"] * 0.02, rel=1e-3)        # rolling: v = w R
    assert o["dissipated_ramp_J"] > 0 and o["dissipated_floor_J"] > 0
    assert o["_ledger_rel"] < 1e-3


def test_2d_3_four_bar():
    o = run("2d_3_four_bar.yaml")
    assert o["rocker_min_deg"] == pytest.approx(87.13, abs=0.1)
    assert o["rocker_max_deg"] == pytest.approx(138.59, abs=0.1)
    assert o["crank_rpm"] == pytest.approx(300 * 12.6 * 0.4 / 30, rel=0.05)
    assert o["_ledger_rel"] < 0.01


def test_3d_1_spinning_top():
    o = run("3d_1_spinning_top.yaml")
    assert o["quarter_turn_s"] == pytest.approx(0.484, rel=0.01)
    assert o["lowest_z_mm"] > 45 and o["_ledger_rel"] < 1e-4


def test_3d_2_drone_arm():
    o = run("3d_2_drone_arm.yaml")
    EI = 70e9 * math.pi * (0.012 ** 4 - 0.010 ** 4) / 64
    P = 8.0 - 0.06 * 9.81
    assert o["tip_up_mm"] == pytest.approx(1000 * P * 0.2 ** 3 / (3 * EI), rel=0.15)
    assert o["root_moment"] == pytest.approx(P * 0.2, rel=0.15)
    assert o["_ledger_rel"] < 1e-3


def test_3d_3_robot_arm():
    o = run("3d_3_robot_arm.yaml")
    assert o["peak_current_A"] == pytest.approx(15.0)
    assert 0.4 < o["t_hit_s"] < 0.9
    assert o["_ledger_rel"] < 0.01


def test_rocket_engine_block_against_published_engines():
    from simblocks.actuators.rocket import rocket_engine, ideal_performance
    f1 = rocket_engine("RP1", 2.27, 7.0e6, 16, mdot=2578.0, Isp_sl=263.0)
    assert f1["Isp_vac"] == pytest.approx(304.0, abs=1.5)        # F-1: 263 s sea level, 304 s vacuum
    r = ideal_performance("RP1", 2.56, 1000 * 6894.757, 9.92)    # textbook LOX / RP-1, 1000 psia, to 1 atm
    assert r["Tc"] == pytest.approx(3676, rel=0.005) and r["Isp_sl"] == pytest.approx(300, rel=0.01)
    h = ideal_performance("LH2", 4.13, 1000 * 6894.757, 8.26)
    assert h["Isp_sl"] == pytest.approx(389.6, rel=0.01)


def test_1d_5_cruise_control():
    o = run("1d_5_cruise_control.yaml")
    assert o["push_hill"] == pytest.approx(o["push_check"], rel=1e-4)    # the PID settles on the force balance
    assert o["v_end"] == pytest.approx(25, rel=1e-4)
    assert 24.0 < o["v_sag"] < 25.0                                       # a brief sag when the hill starts
    assert o["t_on_hill"] == pytest.approx(40, abs=0.01)


def test_1d_6_chua_circuit():
    """Double-scroll chaos: both scrolls visited, the attractor's size as in Kennedy (1992), powers balance."""
    o = run("1d_6_chua_circuit.yaml")
    assert 3.9 < o["v1_max"] < 4.3 and -4.3 < o["v1_min"] < -3.9          # visits both scrolls
    assert 0.7 < o["v2_max"] < 0.85 and 3.2 < o["iL_max"] < 3.8
    assert o["_circuit_ledger_rel"] < 1e-6

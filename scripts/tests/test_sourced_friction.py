"""Tests for the sourced friction / tire / contact / actuator blocks that replace LEGACY ones."""
import math
import numpy as np
import pytest

from simblocks.friction import published as P, laws as L
from simblocks.contact import penalty as C
from simblocks.actuators import motors as M
from simblocks.structures import impact as pf


def test_magic_formula_fit_round_trip():
    f = P.magic_formula_fit(2.0, 1.67, 0.15, 30.0)
    xs = np.linspace(0, 3, 30001)
    y = np.array([P.magic_formula(x, f["B"], f["C"], f["D"], f["E"]) for x in xs])
    assert xs[y.argmax()] == pytest.approx(0.15, abs=2e-4)
    assert y.max() == pytest.approx(2.0, rel=1e-6)
    assert P.magic_formula(1e5, f["B"], f["C"], f["D"], f["E"]) == pytest.approx(1.67, rel=1e-3)
    slope = (P.magic_formula(1e-7, f["B"], f["C"], f["D"], f["E"])) / 1e-7
    assert slope == pytest.approx(30.0, rel=1e-4)


def test_magic_formula_pacejka_dry_road_peak_is_D():
    B, Cc, D, E = 10, 1.9, 1.0, 0.97
    y = max(P.magic_formula(x, B, Cc, D, E) for x in np.linspace(0, 1, 10001))
    assert y == pytest.approx(D, rel=1e-4)


def test_brush_single_mu_closed_form():
    # mu_s = mu_k reduces to Svendenius & Wittenmark 2003: F = mu Fz (1 - (1-u)^3), u = C s/(3 mu Fz)
    Cs, Fz, mu = 3000.0, 100.0, 1.5
    for s in (0.01, 0.05, 0.09):
        u = Cs * s / (3 * mu * Fz)
        assert P.brush_mu(s, Cs, Fz, mu, mu) == pytest.approx(mu * (1 - (1 - u) ** 3), rel=1e-12)
    assert P.brush_mu(1.0, Cs, Fz, mu, mu) == mu


def test_brush_slope_continuity_and_peak_fit():
    Cs, Fz = 3000.0, 100.0
    assert P.brush_mu(1e-8, Cs, Fz, 2.0, 1.67) / 1e-8 == pytest.approx(Cs / Fz, rel=1e-5)
    s0 = 3 * 2.0 * Fz / Cs
    assert P.brush_mu(s0 * (1 - 1e-9), Cs, Fz, 2.0, 1.67) == pytest.approx(1.67, abs=1e-6)
    ms = P.brush_mu_s_for_peak(2.0, 1.67)
    peak = max(P.brush_mu(s, Cs, Fz, ms, 1.67) for s in np.linspace(0, 1, 40001))
    assert peak == pytest.approx(2.0, rel=1e-4)


def test_brush_force_direction_opposes_slip():
    Fx, Fy, sig, mu = P.brush_force_2d(1.0, 0.3, 0.0, 0.02, 10.0, 300.0, 2.0, 1.67, 0.05)
    assert Fx < 0 and Fy < 0 and math.hypot(Fx, Fy) == pytest.approx(mu * 10.0)


def test_relaxation_lag_63_percent_at_one_length_and_holds_at_rest():
    lam, V, dt = 0.02, 2.0, 1e-5
    y = 0.0
    for _ in range(int(round(lam / (V * dt)))):
        y = P.relaxation_lag(y, 1.0, V, dt, lam)
    assert y == pytest.approx(1 - math.exp(-1), rel=1e-6)
    assert P.relaxation_lag(0.3, 1.0, 0.0, 1.0, lam) == pytest.approx(0.3, abs=1e-15)


def test_stribeck_general_reduces_to_legacy_at_delta_2():
    for v in (0.0, 0.01, 0.05, 0.3):
        assert P.stribeck_mu_delta(v, 2.0, 1.67, 0.05, 2.0) == pytest.approx(L.stribeck_mu(v, 2.0, 1.67, 0.05))


def test_lugre_steady_state_matches_stribeck_curve():
    s0, s1, s2, Fc, Fs, vs = 1e5, 316.0, 0.4, 1.0, 1.5, 1e-3
    for v in (5e-4, 2e-3, 0.05):
        z = 0.0
        for _ in range(4000):
            F, z = P.lugre_step(z, v, 1e-4, s0, s1, s2, Fc, Fs, vs)
        g = Fc + (Fs - Fc) * math.exp(-(v / vs) ** 2)
        assert F == pytest.approx(g + s2 * v, rel=1e-6)


def test_lugre_presliding_is_a_spring():
    # tiny displacement, v -> 0: force ~ sigma0 * x (stiff bristle), well below Fs
    z, x, v, dt = 0.0, 0.0, 1e-7, 1e-4
    for _ in range(10):
        F, z = P.lugre_step(z, v, dt, 1e5, 316.0, 0.0, 1.0, 1.5, 1e-3)
        x += v * dt
    assert z == pytest.approx(x, rel=1e-3)


def test_simscape_friction_breakaway_peak():
    vs = np.linspace(1e-4, 0.5, 50001)
    F = np.array([P.simscape_friction(v, 1.5, 1.0, 0.1, 0.0) for v in vs])
    assert F.max() == pytest.approx(1.5, rel=0.03)
    assert F[-1] == pytest.approx(1.0, rel=0.01)


def test_coulomb_tanh_is_simscape_coulomb_term():
    for v in (-0.02, 0.003, 0.4):
        assert L.coulomb_tanh(v, 10.0, 0.5, 0.01) == pytest.approx(-P.simscape_friction(v, 5.0, 5.0, 0.1, 0.0))


def _bounce(k, m, c, v0, dt=1e-7):
    x, v = 0.0, v0
    while True:
        f = max(k * x + c * v, 0.0)
        v -= f / m * dt
        x += v * dt
        if x <= 0.0 or (f == 0.0 and v < 0):
            return -v / v0


def test_restitution_position_vs_force_terminated():
    k, m, e = 1e6, 1.0, 0.5
    c = C.damping_for_restitution(e, k, m)
    e_sp = C.restitution_force_terminated(k, m, c)
    assert e_sp > e
    assert _bounce(k, m, c, 1.0) == pytest.approx(e_sp, rel=5e-3)


def test_flores_damping_hits_target_restitution():
    K, n, m, v0 = 1e9, 1.5, 0.1, 5.0
    for e in (0.3, 0.6, 0.9):
        a = C.hysteresis_damping_factor(e, v0, "flores")
        x, v, dt = 0.0, v0, 1e-8
        while True:
            v -= C.hunt_crossley(x, v, K, n, a) / m * dt
            x += v * dt
            if x <= 0:
                break
        assert -v / v0 == pytest.approx(e, abs=0.06)


def test_matsuki_backdrive_efficiency():
    assert M.backdrive_efficiency(0.8, 20) == pytest.approx(0.750, abs=1e-3)
    assert M.backdrive_efficiency(0.7, 20) == pytest.approx(0.572, abs=1e-3)
    assert M.backdrive_efficiency(0.45, 20) == 0.0
    for ef in np.linspace(0.51, 1.0, 20):
        assert M.backdrive_efficiency(ef, 20) <= ef + 1e-12
    assert M.backdrive_efficiency(1.0, 20) == pytest.approx(1.0)


def test_directional_gearbox():
    assert M.gearbox_output_torque_dir(1.0, 10.0, 20.0, 0.8, 0.75) == pytest.approx(16.0)
    assert M.gearbox_output_torque_dir(-1.0, 10.0, 20.0, 0.8, 0.75) == pytest.approx(-20 / 0.75)


def test_ecm_reduces_to_thevenin_and_relaxes():
    v, i, _ = M.ecm_constant_power(16.8, 0.02, 0.0, 0.0, 0.0, 300.0, 1e-4)
    assert v == pytest.approx(M.bus_voltage(16.8, 0.02, 300.0, 16.8))
    iR1 = 0.0
    for _ in range(int(3 * 0.01 * 1000 / 1e-3)):
        v, i, iR1 = M.ecm_constant_power(16.8, 0.02, 0.01, 1000.0, iR1, 300.0, 1e-3)
    assert iR1 == pytest.approx(i, rel=0.06)     # 3 tau: within e^-3
    assert v < M.bus_voltage(16.8, 0.02, 300.0, 16.8)


def test_challen_oxley_prandtl_limit():
    assert pf.challen_oxley_mu(0.0, 1.0) == pytest.approx(1 / (1 + math.pi / 2))
    assert pf.challen_oxley_mu(0.1, 0.5) > pf.challen_oxley_mu(0.0, 0.5)


def test_blunt_punch():
    F = pf.blunt_punch_line_force(1e9, 1e-3, 6.35e-3)
    assert F == pytest.approx(2 / math.sqrt(3) * (1 + math.pi / 2) * 1e9 * 1e-3 * 6.35e-3)
    assert F / (1e9 * 1e-3 * 6.35e-3) == pytest.approx(2.97, abs=0.01)


def test_axle_transfer_matches_three_point_kernel():
    from simblocks.rigid_body.planar import three_point_normal_loads
    from simblocks.friction.published import axle_load_transfer
    m, g, xc, Lw, h, ax = 1.36, 9.81, 0.03, 0.10, 0.02, 5.0
    NL, NR, NB, _ = three_point_normal_loads(m, g, xc, Lw, 0.12, h, ax, 0.0)
    Wf, Wr = axle_load_transfer(m, g, Lw, xc, h, ax)
    assert NB == pytest.approx(Wf) and NL + NR == pytest.approx(Wr)


def test_no_legacy_without_reason():
    import simblocks as sb
    for b in sb.REGISTRY.values():
        if "LEGACY" in b.source:
            assert ("use " in b.source) or ("reproduction" in b.source) or ("ESC" in b.source), b.name

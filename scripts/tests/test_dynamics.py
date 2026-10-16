"""Contact, constitutive, friction, actuators, rigid body, geometry, integrators, diagnostics."""
import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from _reference import integrate_impact
from simblocks.contact import hertz as H, penalty as PN
from simblocks.constitutive import plasticity as PL, rate_damage as RT
from simblocks.friction import laws as FR
from simblocks.actuators import motors as MO
from simblocks.rigid_body import inertia as IN, planar as PLN
from simblocks.geometry import contact_geometry as G
from simblocks.integrators import explicit as IX
from simblocks.diagnostics.ledger import EnergyLedger, richardson
from simblocks.materials import get, load_curve

rel = lambda a, b: abs(a / b - 1)


# ------------------------------------------------------------------ contact
def test_hertz_impact_vs_ode():
    Es = H.effective_modulus(205e9, 0.29, 205e9, 0.29); Rs = H.effective_radius(0.005); m = 0.05; v = 3.0
    pk = H.hertz_impact(m, v, Es, Rs)
    xm, Fm, tc, vout = integrate_impact(m, v, lambda x, xd: H.hertz_sphere_force(x, Es, Rs), 1e-3)
    assert rel(pk["delta_max"], xm) < 1e-4 and rel(pk["F_max"], Fm) < 1e-3 and rel(pk["t_contact"], tc) < 2e-3
    assert rel(-vout, v) < 1e-5                       # elastic: no energy loss


def test_hertz_pressure_integrates_to_force():
    Es = H.effective_modulus(205e9, 0.29, 70e9, 0.33); Rs = 0.01; F = 500.0
    s = H.hertz_sphere_stress(F, Es, Rs)
    r = np.linspace(0, s["a"], 20001)
    p = s["p0"] * np.sqrt(np.clip(1 - (r / s["a"]) ** 2, 0, None))
    assert rel(np.trapezoid(p * 2 * np.pi * r, r), F) < 1e-4
    assert rel(H.hertz_sphere_force(s["delta"], Es, Rs), F) < 1e-10


@pytest.mark.parametrize("e", [0.2, 0.5, 0.8])
def test_restitution_damping_vs_ode(e):
    k, m = 1e6, 0.1
    c = PN.damping_for_restitution(e, k, m)
    _, _, _, vout = integrate_impact(m, 2.0, lambda x, xd: k * x + c * xd, 1e-2)   # unclipped linear: exact e
    assert rel(-vout / 2.0, e) < 1e-4


def test_collision_impulse_conservation():
    m1, m2, v, e = 1.3, 1.36, 4.0, 0.3
    J, dE = PN.collision_impulse(v, m1, m2, e)
    v1, v2 = v - J / m1, J / m2
    assert rel(m1 * v1 + m2 * v2, m1 * v) < 1e-12 and rel((v2 - v1) / v, e) < 1e-12
    assert rel(0.5 * m1 * v * v - 0.5 * m1 * v1 ** 2 - 0.5 * m2 * v2 ** 2, dE) < 1e-10


def test_spinner_strike_energy_bound():
    s = PN.spinner_strike(7.56e-4, 1257.0, 0.09, 1.36, e=0.0)
    assert 0 < s["E_absorbed"] < s["E_before"] and s["omega_after"] < 1257.0
    # angular momentum: I dw = J R
    assert rel(7.56e-4 * (1257.0 - s["omega_after"]), s["J"] * 0.09) < 1e-12


# ------------------------------------------------------------------ constitutive
def test_tabular_equals_linear_hardening_for_linear_table():
    k, Fy, H_ = 1e5, 100.0, 2e4
    acc_tab = np.array([0.0, 1.0]); F_tab = np.array([Fy, Fy + H_])
    xs = np.concatenate([np.linspace(0, 0.01, 200), np.linspace(0.01, -0.012, 300), np.linspace(-0.012, 0.004, 200)])
    a = (0.0, 0.0); b = (0.0, 0.0, 0.0)
    for x in xs:
        Fa, xp, acc = PL.tabular_return(x, a[0], a[1], k, acc_tab, F_tab, 0); a = (xp, acc)
        Fb, xp2, acc2, bk = PL.linear_hardening_return(x, b[0], b[1], b[2], k, Fy, H_, 0.0); b = (xp2, acc2, bk)
        assert abs(Fa - Fb) < 1e-6 * Fy


def test_tabular_follows_curve_monotonic():
    c = load_curve("7075"); A, Lb = 1e-5, 0.05
    k, acc_tab, F_tab = PL.curve_to_spring_table(c, A, Lb)
    et, st = c.true
    xp = acc = 0.0
    for eps in np.linspace(0, 0.03, 400):
        F, xp, acc = PL.tabular_return(eps * Lb, xp, acc, k, acc_tab, F_tab, 0)
    assert rel(F / A, np.interp(0.03, et, st)) < 5e-3


def test_one_sided_never_pulls_and_unloads_elastically():
    k = 1e5; tab = (np.array([0.0, 1.0]), np.array([50.0, 50.0]))
    xp = acc = 0.0
    F, xp, acc = PL.tabular_return(0.002, xp, acc, k, *tab, 1)
    assert rel(F, 50.0) < 1e-9 and xp > 0
    F2, _, _ = PL.tabular_return(0.002 - 1e-4, xp, acc, k, *tab, 1)
    assert rel(50.0 - F2, k * 1e-4) < 1e-9
    F3, _, _ = PL.tabular_return(-0.001, xp, acc, k, *tab, 1)
    assert F3 == 0.0


def test_plastic_dissipation_positive_cycle():
    k = 1e5; tab = (np.array([0.0, 1.0]), np.array([50.0, 80.0]))
    xs = 0.003 * np.sin(np.linspace(0, 6 * np.pi, 3000))
    xp = acc = 0.0; W = 0.0; Fprev = 0.0
    for i in range(1, len(xs)):
        F, xp, acc = PL.tabular_return(xs[i], xp, acc, k, *tab, 0)
        W += 0.5 * (F + Fprev) * (xs[i] - xs[i - 1]); Fprev = F
    assert W > 0 and acc > 0


def test_lag_return_matches_original_script_lines():
    # verbatim dent update from the legacy hinged-plate sim
    k, kp, y0 = 1e6, 3e5, 1e-4
    sp = 0.0; sb = 0.0
    for pen in np.concatenate([np.linspace(0, 1e-3, 50), np.linspace(1e-3, 0, 20), np.linspace(0, 2e-3, 50)]):
        dyn = y0 + kp / (k - kp) * sp
        ref = sp + (pen - sp - dyn) * (1 - kp / k) if pen - sp > dyn else sp
        sb = PL.lag_return(pen, sb, y0, kp / (k - kp), kp / k, 0)
        assert abs(sb - ref) < 1e-15
        sp = ref


def test_cowper_symonds_and_hyperelastic():
    assert RT.cowper_symonds_factor(0.0, 40.4, 5) == 1.0
    assert rel(RT.cowper_symonds_factor(40.4, 40.4, 5), 2.0) < 1e-12
    # small-strain modulus of neo-Hookean = 3 mu
    mu = 1e6; d = 1e-6
    assert rel(RT.neo_hookean_uniaxial(1 + d, mu) / d, 3 * mu) < 1e-5
    assert rel(RT.mooney_rivlin_uniaxial(1.3, mu / 2, 0.0), RT.neo_hookean_uniaxial(1.3, mu)) < 1e-12


# ------------------------------------------------------------------ friction
def test_tire_curve_peak_and_tail():
    s = np.linspace(1e-4, 3, 30000); mu = np.array([FR.tire_mu(x, 2.0, 1.67, 0.12) for x in s])
    assert 1.9 < mu.max() < 2.2 and abs(mu[-1] - 1.67) < 0.02 and 0.06 < s[mu.argmax()] < 0.2


def test_stribeck_limits_and_tire_force_direction():
    assert rel(FR.stribeck_mu(0.0, 0.25, 0.2, 0.01), 0.25) < 1e-12 and rel(FR.stribeck_mu(1.0, 0.25, 0.2, 0.01), 0.2) < 1e-12
    Fx, Fy, sm, mu = FR.tire_force_2d(1.0, 0.0, 0.0, 0.022, 10.0, 2.0, 1.67, 0.12, 0.05)
    assert Fx < 0 and abs(Fy) < 1e-12 and rel(-Fx, mu * 10.0) < 1e-12


# ------------------------------------------------------------------ actuators
def test_spin_up_time_vs_ode():
    Kt, R, V, Il, J, G_ = 60 / (2 * math.pi * 1000), 0.05, 22.0, 60.0, 7.6e-4, 1.0
    w_t = 0.9 * V / Kt
    t_an = MO.spin_up_time(J, Kt, R, V, Il, w_t, G_)
    def rhs(t, y):
        I = min((V - Kt * y[0]) / R, Il)
        return [Kt * I / J]
    ev = lambda t, y: y[0] - w_t; ev.terminal = True
    sol = solve_ivp(rhs, [0, 100], [0.0], events=ev, rtol=1e-10, atol=1e-10, max_step=0.01)
    assert rel(t_an, sol.t_events[0][0]) < 1e-4


def test_motor_from_curve_reproduces_stall():
    m = MO.motor_from_curve(12.0, 1070, 0.3776, 4.3, 22.6)
    I = MO.motor_current(1.0, 0.0, 12.0, m["Kt"], m["R"], 1e9, 1.0)
    assert rel(MO.gearbox_output_torque(m["Kt"] * I, 1.0, m["G"], m["eta"]), 0.3776) < 1e-9
    assert MO.gearbox_output_torque(1.0, -1.0, 10, 0.8) == pytest.approx(12.5)


def test_bus_voltage_root():
    V = MO.bus_voltage(18.5, 0.05, 300.0, 18.5)
    assert rel(V, 18.5 - 0.05 * 300.0 / V) < 1e-12


# ------------------------------------------------------------------ rigid body, geometry
def test_polygon_plate_vs_primitives():
    a, b, t, rho = 0.1, 0.04, 0.003, 7850
    sq = IN.polygon_plate([(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)], t, rho)
    assert rel(sq["I_cm"], IN.I_rect_plate(rho * t * a * b, a, b)) < 1e-12
    th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    disk = IN.polygon_plate(np.c_[0.09 * np.cos(th), 0.09 * np.sin(th)], t, rho,
                            holes=[np.c_[0.01 * np.cos(th), 0.01 * np.sin(th)]])
    m = rho * t * math.pi * (0.09 ** 2 - 0.01 ** 2)
    assert rel(disk["m"], m) < 1e-5 and rel(disk["I_cm"], IN.I_annulus(m, 0.01, 0.09)) < 1e-5


def test_assembly_parallel_axis():
    asm = IN.assemble([("a", 1.0, -0.1, 0, 0.0), ("b", 1.0, 0.1, 0, 0.0)])
    assert abs(asm["xcom"]) < 1e-15 and rel(asm["Izz"], 0.02) < 1e-12
    m_rem, x_rem = IN.solve_remainder_centroid([("a", 1.0, 0.1, 0, 0)], 2.0, 0.0)
    assert m_rem == 1.0 and rel(x_rem, -0.1) < 1e-12


def test_three_point_loads_sum_and_moments():
    NL, NR, NB, lift = PLN.three_point_normal_loads(1.36, 9.81, 0.065, 0.1, 0.18, 0.023, 0.0, 0.0)
    assert rel(NL + NR + NB, 1.36 * 9.81) < 1e-12 and rel(NB * 0.1, 1.36 * 9.81 * 0.065) < 1e-12 and lift == 0


def test_root_line_arm_is_derivative():
    tx, rx, rz, a, h = 0.07, 0.02, 0.026, 1.0, 1e-7
    p0, arm = G.tip_face_penetration(tx, rx, rz, a)
    dp = (G.tip_face_penetration(tx, rx, rz, a + h)[0] - G.tip_face_penetration(tx, rx, rz, a - h)[0]) / (2 * h)
    assert rel(arm, -dp) < 1e-7


def test_first_contact_sampler_and_doc_table():
    # sampler consistency + doc table value (time-domain MC lives in test_convergence_and_harness)
    rpm, D, v = 9000, 0.2032, 4.0
    d = G.first_contact_delta(v, rpm, D); st = G.first_contact_stats(d)
    R = D / 2; w = rpm * 2 * math.pi / 60; T = 2 * math.pi / w
    rng = np.random.default_rng(1); al = []
    for _ in range(20000):
        u = 1 - d * rng.random()          # face distance at the moment the tip's peak passes = s*/R
        al.append(math.acos(u))
    assert abs(np.mean(al) - st["mean"]) < 0.01 and abs(np.median(al) - st["median"]) < 0.01
    assert max(al) <= st["alpha_max"] + 1e-12
    assert rel(d, 0.263) < 0.01                               # doc table: 8 in, 9000 rpm, 4 m/s


def test_segment_closest():
    cx, cy, s, dist = G.point_segment_closest(2.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    assert (cx, cy, s) == (1.0, 0.0, 1.0) and rel(dist, math.sqrt(2)) < 1e-12


# ------------------------------------------------------------------ integrators, diagnostics
def test_semi_implicit_stability_boundary():
    k, m = 1e6, 1.0; dtc = IX.critical_dt_spring(k, m)
    acc = lambda x, v, t, p: -k * x / m
    for f, bounded in ((0.98, True), (1.02, False)):
        x, v = 1.0, 0.0
        for i in range(4000):
            x, v = IX.semi_implicit_step(acc, x, v, 0, f * dtc, None)
        assert (abs(x) < 10) == bounded


def test_rk4_fourth_order():
    f = lambda s, t, p: np.array([s[1], -s[0]])
    errs = []
    for dt in (0.1, 0.05):
        s = np.array([1.0, 0.0])
        for i in range(int(round(2 / dt))):
            s = IX.rk4_step(f, s, i * dt, dt, None)
        errs.append(abs(s[0] - math.cos(2)))
    assert 14 < errs[0] / errs[1] < 18


def test_ledger_and_richardson():
    L = EnergyLedger(); L.add("motor", 10.0, source=True); L.add("heat", 4.0)
    assert L.close(0.0, 6.0)["ok"] and not L.close(0.0, 7.0)["ok"]
    assert rel(richardson(1.1, 1.05, order=1), 1.0) < 1e-12

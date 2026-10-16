"""(1) Fixed-step dt convergence of every time-integrated kernel, (2) an independent time-domain Monte Carlo
for the first-contact distribution, (3) the verify_all harness must CATCH planted bugs."""
import math
import numpy as np
import pytest

from simblocks.contact import hertz as H, penalty as PN
from simblocks.constitutive import plasticity as PL
from simblocks.friction import laws as FR
from simblocks.thermal import basic as TH
from simblocks.geometry import contact_geometry as G
from simblocks.diagnostics.verify import SimSpec, verify_all, PASS, FAIL, WARN
from simblocks.integrators.explicit import critical_dt_spring

rel = lambda a, b: abs(a / b - 1)


# ------------------------------------------------------------------ toy sim built from blocks
def dent_impact(p, dt):
    """Mass m at v0 hits a one-sided elastic-plastic dent (tabular) with an optional Hertz tip in series
    approximation (here: dent only) and an optional dashpot. Semi-implicit Euler. Full energy ledger."""
    m, v0, k, Fy, H_ = p["m"], p["v0"], p["k"], p["Fy"], p["H"]
    c = p.get("c", 0.0); bug = p.get("bug_energy", 0.0)
    acc_tab = np.array([0.0, 1.0]); F_tab = np.array([Fy, Fy + H_])
    x = 0.0; v = v0; xp = acc = 0.0; W_damp = 0.0; W_plastic = 0.0; Fmax = 0.0
    t = 0.0; F_prev = 0.0; xp_prev = 0.0
    n = int(round(p["t_end"] / dt))
    for i in range(n):
        F, xp, acc = PL.tabular_return(x, xp, acc, k, acc_tab, F_tab, 1)
        Fd = c * v if F > 0 else 0.0
        Ftot = F + Fd + bug * v * v          # 'bug' injects a fake force to test the ledger veto
        v -= Ftot / m * dt
        W_plastic += F * (xp - xp_prev); xp_prev = xp
        W_damp += Fd * v * dt
        x += v * dt
        Fmax = max(Fmax, F)
    Fend, _, _ = PL.tabular_return(x, xp, acc, k, acc_tab, F_tab, 1)
    E_el = 0.5 * Fend * Fend / k if Fend > 0 else 0.0
    E0 = 0.5 * m * v0 * v0; E1 = 0.5 * m * v * v + E_el
    res = E0 - E1 - W_plastic - W_damp
    return dict(set_mm=xp * 1e3, rebound=-v, Fmax=Fmax, _ledger_rel=res / max(E0, 1e-30), unused=p.get("unused", 1.0) * 0 + 1.0)


P0 = dict(m=1.3, v0=3.0, k=5e6, Fy=2000.0, H=2e5, c=0.0, t_end=4e-3)
CRIT = lambda p: critical_dt_spring(p["k"], p["m"])


def make_spec(**kw):
    d = dict(name="dent impact toy", run=dent_impact, params=dict(P0), dt=1e-6,
             metrics=["set_mm", "rebound", "Fmax"],
             uncertain={"v0": (2.0, 4.0), "Fy": (1500.0, 2500.0), "k": (3e6, 8e6)},
             knockouts={"plastic_off": (dict(Fy=1e12), {"set_mm": "-", "rebound": "+"}),
                        "damping_on": (dict(c=50.0), {"rebound": "-"})},
             limits=[("zero speed -> zero set", dict(v0=0.0), {"set_mm": 0.0}),
                     ("elastic limit -> full rebound", dict(Fy=1e12), {"rebound": lambda r: abs(r - 3.0) < 3e-3})],
             bounds={"set_mm": (0.0, 50.0), "rebound": (-1.0, 3.0001)},
             critical_dt=CRIT, mc_n=60, conv_tol=0.01)
    d.update(kw)
    return SimSpec(**d)


def status(rep, name):
    return next(c for c in rep.checks if c.name == name).status


def test_harness_passes_correct_sim():
    rep = verify_all(make_spec())
    assert rep.ok, rep.markdown()
    for n in ("determinism", "finite_outputs", "physical_bounds", "stability_margin", "energy_ledger",
              "knockout:plastic_off", "knockout:damping_on", "monte_carlo"):
        assert status(rep, n) == PASS, n


def test_harness_catches_energy_bug():
    rep = verify_all(make_spec(params={**P0, "bug_energy": -300.0}), skip=("monte_carlo", "sensitivity"))
    assert status(rep, "energy_ledger") == FAIL and not rep.ok


def test_harness_catches_dead_knockout():
    rep = verify_all(make_spec(knockouts={"does_nothing": dict(unused=5.0)}), skip=("monte_carlo", "sensitivity"))
    assert status(rep, "knockout:does_nothing") == FAIL


def test_harness_catches_wrong_sign_knockout():
    rep = verify_all(make_spec(knockouts={"damping_on": (dict(c=50.0), {"rebound": "+"})}),
                     skip=("monte_carlo", "sensitivity"))
    assert status(rep, "knockout:damping_on") == FAIL


def test_harness_catches_unconverged_dt():
    rep = verify_all(make_spec(dt=4e-4), skip=("monte_carlo", "sensitivity"))
    assert status(rep, "dt_convergence") == FAIL


def test_harness_catches_unstable_dt():
    rep = verify_all(make_spec(dt=2.5 * CRIT(P0)), skip=("monte_carlo", "sensitivity", "convergence"))
    assert status(rep, "stability_margin") == FAIL


def test_harness_catches_bad_limit():
    rep = verify_all(make_spec(limits=[("wrong expectation", dict(v0=0.0), {"set_mm": 1.0})]),
                     skip=("monte_carlo", "sensitivity"))
    assert status(rep, "limit:wrong expectation") == FAIL


def test_harness_no_monotonicity_judgement():
    rep = verify_all(make_spec(), skip=("monte_carlo",))
    assert not any("monoton" in c.name for c in rep.checks)
    assert status(rep, "sensitivity") == PASS


# ------------------------------------------------------------------ fixed-step convergence of kernels
def _hertz_fixed(dt, m=0.05, v=3.0):
    Es = H.effective_modulus(205e9, 0.29, 205e9, 0.29); Rs = 0.005
    x, u, Fm = 0.0, v, 0.0
    while True:
        F = H.hertz_sphere_force(x, Es, Rs); Fm = max(Fm, F)
        u -= F / m * dt; x += u * dt
        if x < 0 and u < 0:
            break
    return Fm, -u, H.hertz_impact(m, v, Es, Rs)["F_max"]


def test_hertz_fixed_step_converges_to_exact():
    errs = [abs(_hertz_fixed(dt)[0] / _hertz_fixed(dt)[2] - 1) for dt in (2e-6, 1e-6, 5e-7, 2.5e-7)]
    assert errs[-1] < 1e-3 and errs[-1] < errs[0]
    assert rel(_hertz_fixed(2.5e-7)[1], 3.0) < 1e-3        # elastic rebound


def test_tire_spinup_fixed_step_converges():
    # wheel with torque on a mass through tire_force_2d, compare dt and dt/4 after 0.3 s
    def run(dt):
        m, J, rw, N, T = 1.36, 1e-4, 0.022, 13.3, 0.15
        v = w = 0.0
        for _ in range(int(0.3 / dt)):
            Fx, _, _, _ = FR.tire_force_2d(v, 0.0, w, rw, N, 2.0, 1.67, 0.12, 0.05)
            w += (T + Fx * rw) / J * dt; v -= Fx / m * dt
        return v
    v1, v2, v3 = run(1e-4), run(5e-5), run(2.5e-5)
    assert abs(v2 - v3) < abs(v1 - v2) and rel(v1, v3) < 0.01


def test_thermal_node_fixed_step_vs_analytic():
    m, cp, h, A, T0, Ta, Q = 0.05, 900.0, 15.0, 0.01, 300.0, 300.0, 20.0
    for dt, tol in ((1.0, 2e-2), (0.1, 2e-3)):
        T = T0
        for _ in range(int(600 / dt)):
            T += TH.node_dTdt(T, Q, Ta, 1 / (h * A), m, cp) * dt
        exact = TH.lumped_temperature(600, T0, Ta, h, A, m, cp, Q)
        assert abs(T - exact) / (exact - T0) < tol


# ------------------------------------------------------------------ independent first-contact Monte Carlo
def test_first_contact_time_domain_monte_carlo():
    """Simulate the actual rotating tip and closing face (no use of the closed form), compare to the doc."""
    rpm, D, v = 9000, 0.2032, 4.0
    R = D / 2; w = rpm * 2 * math.pi / 60
    st = G.first_contact_stats(G.first_contact_delta(v, rpm, D))
    rng = np.random.default_rng(3); n = 4000
    phi0 = rng.uniform(-math.pi, math.pi, n)
    s0 = R * 1.01
    dt = (2 * math.pi / w) / 4000
    t = np.zeros(n); alpha = np.full(n, np.nan); active = np.ones(n, bool)
    for i in range(40000):
        tt = i * dt
        phi = phi0 + w * tt
        hit = active & (R * np.cos(phi) >= s0 - v * tt)
        if hit.any():
            a = -(np.mod(phi[hit] + math.pi, 2 * math.pi) - math.pi)    # angle before top-dead-center
            alpha[hit] = a; active[hit] = False
        if not active.any():
            break
    assert not np.isnan(alpha).any()
    # trailing-side hits exist but only within sin(phi) < v/(w R): the doc's "never trailing" holds to that band
    assert alpha.min() >= -1.2 * v / (w * R)
    assert np.mean(alpha < 0) < 0.05 and alpha.max() <= st["alpha_max"] + 0.01
    # doc: exact contact lands a few degrees earlier in the swing than the closed form (tip still advancing)
    assert 0.85 * st["mean"] < np.mean(alpha) < 1.01 * st["mean"]


# ------------------------------------------------------------------ adaptive time step
from simblocks.integrators.adaptive import adaptive_dt, stable_step


def flight_then_dent(p, dt):
    """Mass flies a long gap, hits a one-sided elastic-plastic dent, rebounds. dt = refinement scale.
    Adaptive mode scales dt_min, dt_max and C with dt; fixed mode uses dt directly."""
    m, v0, k, Fy, H_, gap0 = p["m"], p["v0"], p["k"], p["Fy"], p["H"], p["gap"]
    acc_tab = np.array([0.0, 1.0]); F_tab = np.array([Fy, Fy + H_])
    s = dt / p["dt_ref"]
    x = -gap0; v = v0; xp = acc = 0.0; W = 0.0; t = 0.0; steps = 0
    h = p["dt_max"] * s if p["adaptive"] else dt
    while t < p["t_end"]:
        if p["adaptive"]:
            dst = stable_step(k, m, p["C"] * s)
            h = adaptive_dt(h, dst, x > 0.0, max(-x, 0.0), v, p["dt_min"] * s, p["dt_max"] * s, 20.0, 1.2)
        xpo = xp
        F, xp, acc = PL.tabular_return(x, xp, acc, k, acc_tab, F_tab, 1)
        W += F * (xp - xpo)
        v -= F / m * h; x += v * h; t += h; steps += 1
    F, _, _ = PL.tabular_return(x, xp, acc, k, acc_tab, F_tab, 1)
    E0 = 0.5 * m * v0 ** 2; E1 = 0.5 * m * v * v + (0.5 * F * F / k if F > 0 else 0.0)
    return dict(set_mm=xp * 1e3, rebound=-v, steps=steps, _ledger_rel=(E0 - E1 - W) / E0)


PA = dict(m=1.3, v0=3.0, k=5e6, Fy=2000.0, H=2e5, gap=0.3, t_end=0.2, dt_ref=1e-6,
          C=0.05, dt_min=1e-8, dt_max=2e-4, adaptive=True)


def test_adaptive_matches_fixed_fine_step_with_far_fewer_steps():
    fixed = flight_then_dent(dict(PA, adaptive=False), 1e-6)
    adapt = flight_then_dent(PA, 1e-6)
    assert rel(adapt["set_mm"], fixed["set_mm"]) < 5e-3 and rel(adapt["rebound"], fixed["rebound"]) < 5e-3
    assert adapt["steps"] < fixed["steps"] / 20
    assert abs(adapt["_ledger_rel"]) < 0.01


def test_adaptive_never_leaves_caps_and_shrinks_before_contact():
    h = 2e-4
    h = adaptive_dt(h, 1e-6, False, 1e-3, 3.0, 1e-8, 2e-4, 20.0, 1.2)       # 1 mm gap closing at 3 m/s
    assert h <= 1e-3 / (20 * 3.0) + 1e-15
    assert adaptive_dt(1e-6, 1e-6, False, 1.0, -1.0, 1e-8, 2e-4, 20.0, 1.2) == pytest.approx(1.2e-6)   # smooth growth
    assert adaptive_dt(1.0, 1e-12, True, 0.0, 0.0, 1e-8, 2e-4, 20.0, 1.2) == 1e-8                     # floor
    assert adaptive_dt(1e-3, 1.0, False, 1.0, -1.0, 1e-8, 2e-4, 20.0, 1.2) == 2e-4                    # ceiling


def test_adaptive_sim_passes_harness():
    spec = SimSpec(name="adaptive flight + dent", run=flight_then_dent, params=dict(PA), dt=1e-6,
                   metrics=["set_mm", "rebound"], uncertain={"v0": (2.0, 4.0)}, mc_n=20,
                   knockouts={"no plasticity": (dict(Fy=1e12), {"set_mm": "-"})},
                   limits=[("no speed", dict(v0=1e-9), {"set_mm": 0.0})], bounds={"set_mm": (0, 50)})
    rep = verify_all(spec, skip=("sensitivity",))
    assert rep.ok, rep.markdown()

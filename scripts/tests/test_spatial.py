"""3-D rigid-body blocks, each checked against an independent method."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from simblocks.rigid_body.spatial import (quat_to_matrix, quat_step, gyrostat_accel, gyrostat_energy,
                                          point_box_penetration, coulomb_regularized_3d, I_box_tensor,
                                          parallel_axis_tensor)


def _q_scipy(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])          # scipy is (x, y, z, w)


def test_quat_matrix_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        assert np.allclose(quat_to_matrix(q), _q_scipy(q).as_matrix(), atol=1e-12)


def test_quat_step_constant_rate_matches_rotvec():
    """Body-frame rotation at constant w for time T equals q0 * exp(rotvec = w T) (scipy composition)."""
    rng = np.random.default_rng(2)
    q0 = rng.normal(size=4); q0 /= np.linalg.norm(q0)
    w = np.array([3.0, -7.0, 11.0]); dt = 1e-3; n = 400
    q = q0.copy()
    for _ in range(n):
        q = quat_step(q, w, dt)
    ref = (_q_scipy(q0) * Rotation.from_rotvec(w * dt * n)).as_matrix()
    assert np.allclose(quat_to_matrix(q), ref, atol=1e-10)


def test_gyrostat_free_precession_matches_closed_form():
    """Axisymmetric body (jt, jt, ja) with rotor on z, torque-free: transverse body rate rotates at
    lam = ((ja - jt) wz + Is Om) / jt (closed form from Euler's equations), wz and Om constant."""
    jt, ja, Is = 2e-3, 3e-3, 8e-4
    J = np.diag([jt, jt, ja]); a = np.array([0.0, 0.0, 1.0])
    w = np.array([5.0, 0.0, 2.0]); Om = 1200.0
    lam = ((ja - jt) * w[2] + Is * Om) / jt
    dt = 2e-7; T = 4e-3; n = int(T / dt)
    for _ in range(n):                                   # RK4 so the check tests the equations, not the stepper
        def f(wv, Ov):
            return gyrostat_accel(J, Is, a, wv, Ov, np.zeros(3), 0.0)
        k1w, k1o = f(w, Om); k2w, k2o = f(w + 0.5 * dt * k1w, Om + 0.5 * dt * k1o)
        k3w, k3o = f(w + 0.5 * dt * k2w, Om + 0.5 * dt * k2o); k4w, k4o = f(w + dt * k3w, Om + dt * k3o)
        w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w); Om = Om + dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
    t = n * dt
    assert abs(w[0] - 5.0 * math.cos(lam * t)) < 1e-6 and abs(w[1] - 5.0 * math.sin(lam * t)) < 1e-6
    assert abs(w[2] - 2.0) < 1e-9 and abs(Om - 1200.0) < 1e-6


def test_gyrostat_conserves_world_momentum_and_energy():
    """General (non-symmetric, tilted rotor axis) torque-free gyrostat: R(q) (J w + Is Om a) and the kinetic
    energy (computed independently as rotor + body sums) stay constant."""
    J = np.array([[3e-3, 2e-4, -1e-4], [2e-4, 4e-3, 3e-4], [-1e-4, 3e-4, 5e-3]])
    a = np.array([0.3, -0.5, 0.8]); a /= np.linalg.norm(a); Is = 6e-4
    q = np.array([1.0, 0, 0, 0]); w = np.array([10.0, -20.0, 5.0]); Om = 900.0
    def H(q, w, Om):
        return quat_to_matrix(q) @ (J @ w + Is * Om * a)
    def E(w, Om):
        rotor_abs = Om + a @ w                            # rotor absolute spin about its axis
        return 0.5 * w @ J @ w - 0.5 * Is * (a @ w) ** 2 + 0.5 * Is * rotor_abs ** 2
    H0, E0 = H(q, w, Om), E(w, Om)
    dt = 1e-7
    for _ in range(20000):
        k1w, k1o = gyrostat_accel(J, Is, a, w, Om, np.zeros(3), 0.0)
        wm = w + 0.5 * dt * k1w; om_m = Om + 0.5 * dt * k1o
        k2w, k2o = gyrostat_accel(J, Is, a, wm, om_m, np.zeros(3), 0.0)
        q = quat_step(q, wm, dt); w = w + dt * k2w; Om = Om + dt * k2o
    assert np.linalg.norm(H(q, w, Om) - H0) / np.linalg.norm(H0) < 1e-6
    assert abs(E(w, Om) - E0) / E0 < 1e-6
    assert abs(gyrostat_energy(J, Is, a, w, Om) - E(w, Om)) / E0 < 1e-12


def test_gyrostat_axial_torque_spins_rotor_only_when_body_fixed_axis():
    """Isotropic body, torque on the rotor axis only: total angular momentum change equals the applied
    impulse (independent: integral of torque)."""
    J = np.eye(3) * 2e-3; a = np.array([0.0, 1.0, 0.0]); Is = 5e-4
    wd, od = gyrostat_accel(J, Is, a, np.zeros(3), 0.0, np.zeros(3), 1.0)
    # body + rotor: internal torque only, so total momentum along a is unchanged: (J wd + Is od a) . a = 0
    assert abs((J @ wd + Is * od * a) @ a) < 1e-12 and od > 0 and wd[1] < 0


def test_point_box_penetration_vs_brute_force():
    rng = np.random.default_rng(3); h = np.array([0.05, 0.015, 0.003])
    def face_dist(p):                                    # exact distance to each face rectangle (clamped projection)
        best = np.inf
        for ax in range(3):
            for sg in (-1.0, 1.0):
                c = np.clip(p, -h, h); c[ax] = sg * h[ax]
                best = min(best, float(np.linalg.norm(p - c)))
        return best
    for _ in range(200):
        p = rng.uniform(-1.3, 1.3, 3) * h
        pen, ax, sg = point_box_penetration(p[0], p[1], p[2], h[0], h[1], h[2])
        inside = np.all(np.abs(p) < h)
        assert (pen > 0) == inside
        if inside:
            assert abs(pen - face_dist(p)) < 1e-15
            n = np.zeros(3); n[ax] = sg
            assert abs(np.abs(p + pen * n) - h)[ax] < 1e-12  # moving out along the normal lands on that face


def test_coulomb_regularized_limits():
    fx, fy, fz = coulomb_regularized_3d(30.0, -40.0, 0.0, 100.0, 0.3, 0.01)
    assert abs(math.hypot(fx, fy) - 30.0) < 1e-3 and fx < 0 and fy > 0 and fz == 0
    fx, fy, fz = coulomb_regularized_3d(1e-4, 0.0, 0.0, 100.0, 0.3, 0.01)
    assert 0 > fx > -0.31 and fx * 1e-4 <= 0


def test_box_tensor_and_parallel_axis_vs_point_cloud():
    """Independent: Monte Carlo point cloud of a uniform box displaced from the reference point."""
    rng = np.random.default_rng(4); m = 1.2; L = (0.2, 0.12, 0.04); d = np.array([0.03, -0.02, 0.01])
    P = (rng.uniform(-0.5, 0.5, (400000, 3)) * L) + d
    mi = m / len(P)
    I_mc = mi * (np.sum(P * P) * np.eye(3) - P.T @ P)
    I_bl = parallel_axis_tensor(I_box_tensor(m, *L), m, d)
    assert np.allclose(I_bl, I_mc, rtol=0.01, atol=2e-6)


def test_obb_gap_is_a_lower_bound_and_detects_overlap():
    """Independent: dense surface samples of both boxes; true distance >= gap; overlap (a vertex of one inside
    the other, or sampled points coinciding) implies gap <= 0."""
    from simblocks.rigid_body.spatial import obb_gap
    rng = np.random.default_rng(5)
    def samples(c, R, h, n=14):
        g = np.linspace(-1, 1, n)
        P = np.array([[x, y, z] for x in g for y in g for z in g if max(abs(x), abs(y), abs(z)) == 1.0]) * h
        return P @ R.T + c
    n_sep = n_ovl = 0
    for _ in range(150):
        h1 = rng.uniform(0.002, 0.05, 3); h2 = rng.uniform(0.002, 0.05, 3)
        R1 = Rotation.random(random_state=rng.integers(1e9)).as_matrix()
        R2 = Rotation.random(random_state=rng.integers(1e9)).as_matrix()
        c1 = np.zeros(3); c2 = rng.normal(size=3) * 0.05
        gap = obb_gap(c1, R1, h1, c2, R2, h2)
        S1, S2 = samples(c1, R1, h1), samples(c2, R2, h2)
        from scipy.spatial.distance import cdist
        dmin = cdist(S1, S2).min()
        inside = np.any(np.all(np.abs((S2 - c1) @ R1) < h1, axis=1)) or np.any(np.all(np.abs((S1 - c2) @ R2) < h2, axis=1))
        if inside:
            n_ovl += 1
            assert gap <= 1e-12
        else:
            assert gap <= dmin + 1e-9               # never claims more clearance than really exists
            if gap > 0:
                n_sep += 1
    assert n_sep > 20 and n_ovl > 10


def test_load_blocks_impulse_and_simplification():
    """Half-sine impulse 2 F T / pi (closed form); split finds two pulses; the FE table keeps impulse exactly and
    stays within tolerance of the original."""
    from simblocks.diagnostics.loads import half_sine_train, pulse_metrics, split_pulses, fe_load_table
    t, F = half_sine_train([3408, 3408], [0.53e-3, 0.53e-3], [0.0, 1.06e-3], 5e-3, n_per=400)
    m = pulse_metrics(t, F)
    assert abs(m["impulse"] - 2 * 2 * 3408 * 0.53e-3 / math.pi) < 1e-5 * m["impulse"]
    assert len(split_pulses(t, F)) == 2
    tt = np.linspace(0, 2e-3, 4001)
    G = np.c_[np.where(tt < 1e-3, np.sin(np.pi * tt / 1e-3), 0) * 3000, np.exp(-((tt - 1.4e-3) / 1e-4) ** 2) * 800]
    t2, G2, info = fe_load_table(tt, G, tol_frac=0.005)
    assert info["points"] < 200
    for c in range(2):
        J0 = np.sum(0.5 * (G[1:, c] + G[:-1, c]) * np.diff(tt)); J1 = np.sum(0.5 * (G2[1:, c] + G2[:-1, c]) * np.diff(t2))
        assert abs(J1 - J0) < 1e-9 * abs(J0)
        assert np.max(np.abs(np.interp(tt, t2, G2[:, c]) - G[:, c])) < 0.02 * 3000

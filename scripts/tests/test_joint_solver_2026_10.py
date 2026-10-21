"""The faster rigid-joint solve (2026-10-20): the small dense solvers are right, and the worked examples give the
same answers as the old solver (reference values recorded from it before the rebuild)."""
import os
import numpy as np
import pytest
from simblocks.compiler import load_system
from simblocks.compiler.linsolve import _gauss, _lu, _lu_solve

HERE = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.join(HERE, "..", "examples", "learn")


@pytest.mark.parametrize("n", [1, 3, 10, 24])
def test_small_solvers_match_numpy(n):
    rng = np.random.default_rng(n)
    K = rng.normal(size=(n, n)) + n * np.eye(n)
    b = rng.normal(size=n)
    ref = np.linalg.solve(K, b)
    x = b.copy(); Kc = K.copy()
    assert _gauss(Kc, x, n)
    assert np.allclose(x, ref, rtol=1e-12, atol=1e-12)
    Kl = K.copy(); piv = np.zeros(n, np.int64); y = b.copy()
    assert _lu(Kl, piv, n)
    _lu_solve(Kl, piv, y, n)
    assert np.allclose(y, ref, rtol=1e-12, atol=1e-12)


def test_singular_is_reported():
    K = np.array([[1.0, 2.0], [2.0, 4.0]]); x = np.array([1.0, 1.0])
    assert not _gauss(K.copy(), x, 2)
    assert not _lu(K.copy(), np.zeros(2, np.int64), 2)


# outputs of the solver before the rebuild (16 Gauss-Seidel sweeps + row clean-ups every step)
OLD = {
    "3d_3_robot_arm.yaml": {"t_hit_s": 0.6226003862990178, "stop_force_N": 369.69593413424684,
                            "stop_dissipated_J": 0.173920581947862, "bumper_mm": 1.7571195368962216,
                            "shoulder_force_N": 325.68010223474107, "payload_top_mm": 819.1792451984807},
    "../../tests/fixtures/top_on_ball_joint.yaml": {"quarter_turn_s": 0.48400959906224783, "lowest_z_mm": 46.98403573137952,
                               "tip_force_N": 3.0981405013598073},
    "2d_3_four_bar.yaml": {"rocker_min_deg": 87.13401769212601, "rocker_max_deg": 138.59037783394845,
                           "crank_rpm": 50.208721388842335, "force_A_N": 9.825175797417984,
                           "force_O4_N": 3.5836270467815883},
    "2d_1_pendulum.yaml": {"quarter_period_s": 0.2944915092006668, "max_speed_rad_s": 2.808159943931987,
                           "hinge_force_N": 3.180988894927531},
}


@pytest.mark.parametrize("name", sorted(OLD))
def test_examples_match_the_old_solver(name):
    out = load_system(open(os.path.join(LEARN, name)).read()).run()
    for k, v in OLD[name].items():
        assert out[k] == pytest.approx(v, rel=1e-6), k

import math
import numpy as np
import pytest

from _reference import fe_beam, fe_beam_uniform_loads, axisym_plate_uniform, rotating_disk_bvp
from simblocks.materials import get
from simblocks.structures.sections import properties as S
from simblocks.structures.beams import static as B
from simblocks.structures.plates import circular as PC, rectangular as PR, yield_line as YL
from simblocks.structures import rotating as RD
from simblocks.structures import shells as SH
from simblocks.structures import torsion as TS
from simblocks.structures import buckling as BK
from simblocks.structures import joints as FJ
from simblocks.structures import vibration as VM

STEEL = get("4130_45HRC")
SEC = S.rectangle(0.02, 0.004)
L = 0.1
EI = STEEL.E * SEC["Ix"]
N = 40
rel = lambda a, b: abs(a / b - 1)


# ------------------------------------------------------------------ sections
def test_rectangle_vs_polygon():
    p = S.polygon_section([(0, 0), (0.02, 0), (0.02, 0.004), (0, 0.004)])
    for k in ("A", "Ix", "Iy", "S"):
        assert rel(p[k], SEC[k]) < 1e-9
    assert rel(p["Z"], SEC["Z"]) < 2e-3


def test_rect_torsion_vs_saint_venant_series():
    for b, h in ((0.02, 0.004), (0.01, 0.01), (0.03, 0.01)):
        A, Bh = max(b, h), min(b, h)
        s = sum(math.tanh(n * math.pi * A / (2 * Bh)) / n ** 5 for n in range(1, 60, 2))
        J = A * Bh ** 3 / 3 * (1 - 192 / math.pi ** 5 * Bh / A * s)
        assert rel(S.rectangle(b, h)["J"], J) < 0.005


def test_tube_and_circle_vs_polygon():
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circ = [(0.01 * math.cos(t), 0.01 * math.sin(t)) for t in th]
    p = S.polygon_section(circ); c = S.circle(0.02)
    assert rel(p["Ix"], c["Ix"]) < 1e-4 and rel(p["Z"], c["Z"]) < 2e-3


def test_i_beam_vs_polygon():
    bf, tf, hw, tw = 0.05, 0.005, 0.04, 0.004
    h = hw + 2 * tf
    pts = [(-bf/2, -h/2), (bf/2, -h/2), (bf/2, -h/2+tf), (tw/2, -h/2+tf), (tw/2, h/2-tf), (bf/2, h/2-tf),
           (bf/2, h/2), (-bf/2, h/2), (-bf/2, h/2-tf), (-tw/2, h/2-tf), (-tw/2, -h/2+tf), (-bf/2, -h/2+tf)]
    p = S.polygon_section(pts); s = S.i_beam(bf, tf, hw, tw)
    assert rel(p["Ix"], s["Ix"]) < 1e-9 and rel(p["Z"], s["Z"]) < 3e-3


def test_shape_factors():
    assert rel(S.shape_factor(SEC), 1.5) < 1e-12
    assert rel(S.shape_factor(S.circle(0.01)), 16 / (3 * math.pi)) < 1e-12


# ------------------------------------------------------------------ beams vs FE
@pytest.mark.parametrize("case", ["cant_end", "cant_a", "ss_center", "ss_a", "ff_center", "fg_end", "prop_center"])
def test_beam_point_cases_vs_fe(case):
    mid = N // 2; a_node = 3 * N // 10; a = a_node * L / N
    spec = {
        "cant_end": (B.cantilever_end_load(STEEL, SEC, L), {0: [0, 1]}, N),
        "cant_a": (B.cantilever_point_at(STEEL, SEC, L, a), {0: [0, 1]}, a_node),
        "ss_center": (B.simply_supported_center(STEEL, SEC, L), {0: [0], N: [0]}, mid),
        "ss_a": (B.simply_supported_point_at(STEEL, SEC, L, a), {0: [0], N: [0]}, a_node),
        "ff_center": (B.fixed_fixed_center(STEEL, SEC, L), {0: [0, 1], N: [0, 1]}, mid),
        "fg_end": (B.fixed_guided_end_load(STEEL, SEC, L), {0: [0, 1], N: [1]}, N),
        "prop_center": (B.propped_center(STEEL, SEC, L), {0: [0, 1], N: [0]}, mid),
    }[case]
    blk, bcs, node = spec
    fe = fe_beam(L, EI, N, bcs, [(node, 0, 1.0)])
    assert rel(blk["k"], 1 / fe["w"][node]) < 1e-6
    assert rel(blk["M_max"], fe["Mmax"]) < 1e-6


@pytest.mark.parametrize("case", ["cant", "ss", "ff", "prop"])
def test_beam_uniform_cases_vs_fe(case):
    ne = 200 if case == "prop" else N
    bcs = {"cant": {0: [0, 1]}, "ss": {0: [0], ne: [0]}, "ff": {0: [0, 1], ne: [0, 1]}, "prop": {0: [0, 1], ne: [0]}}[case]
    blk = {"cant": B.cantilever_uniform, "ss": B.simply_supported_uniform, "ff": B.fixed_fixed_uniform,
           "prop": B.propped_uniform}[case](STEEL, SEC, L)
    fe = fe_beam(L, EI, ne, bcs, fe_beam_uniform_loads(L, ne, 1.0))
    w = fe["w"][-1] if case == "cant" else (fe["w"].max() if case == "prop" else fe["w"][len(fe["w"]) // 2])
    assert rel(blk["delta"], w) < 2e-3
    assert rel(blk["M_max"], fe["Mmax"]) < 1e-3


def test_timoshenko_adds_shear_compliance():
    deep = S.rectangle(0.02, 0.05)
    eb = B.cantilever_end_load(STEEL, deep, 0.1); tm = B.cantilever_end_load(STEEL, deep, 0.1, timoshenko=True)
    extra = tm["delta"] - eb["delta"]
    assert rel(extra, 0.1 / (deep["kappa"] * STEEL.G * deep["A"])) < 1e-12


def test_collapse_loads_virtual_work():
    Mp = STEEL.sy * SEC["Z"]
    # mechanism virtual work, independent of the block's coefficients
    assert rel(B.cantilever_end_load(STEEL, SEC, L)["F_c"], Mp / L) < 1e-12
    assert rel(B.simply_supported_center(STEEL, SEC, L)["F_c"], 2 * Mp * 2 / L) < 1e-12        # P d = Mp 2 theta
    assert rel(B.fixed_fixed_center(STEEL, SEC, L)["F_c"], Mp * 4 * 2 / L) < 1e-12           # 4 hinges-rotations
    assert rel(B.fixed_guided_end_load(STEEL, SEC, L)["F_c"], 2 * Mp / L) < 1e-12
    assert rel(B.propped_center(STEEL, SEC, L)["F_c"], 3 * Mp * 2 / L) < 1e-12
    # propped uniform: hinges at the fixed end and at x from it; W(x) = 2 Mp (2/x + 1/(L-x)), minimized
    xs = np.linspace(0.3, 0.8, 50001) * L
    W = 2 * Mp * (2 / xs + 1 / (L - xs))
    assert rel(B.propped_uniform(STEEL, SEC, L)["F_c"], W.min()) < 1e-4


def test_springs():
    assert rel(B.springs_in_series(100, 100), 50) < 1e-12 and B.springs_in_parallel(1, 2) == 3


# ------------------------------------------------------------------ plates
def test_circular_uniform_vs_bvp():
    a, t = 0.05, 0.002
    D = PC.flexural_rigidity(STEEL.E, t, STEEL.nu)
    assert rel(PC.circ_clamped_pressure(STEEL, a, t)["w_per_q"], axisym_plate_uniform(a, D, STEEL.nu, 1.0, True)) < 1e-6
    assert rel(PC.circ_ss_pressure(STEEL, a, t)["w_per_q"], axisym_plate_uniform(a, D, STEEL.nu, 1.0, False)) < 1e-6


def test_circular_point_vs_annular_limit():
    a, t = 0.05, 0.002
    D = PC.flexural_rigidity(STEEL.E, t, STEEL.nu)
    w_cl = PC._annular_inner_line_deflection(a, 1e-7, STEEL.nu, D, fixed_outer=True)
    w_ss = PC._annular_inner_line_deflection(a, 1e-7, STEEL.nu, D, fixed_outer=False)
    assert rel(PC.circ_clamped_center_point(STEEL, a, t)["k"], 1 / w_cl) < 1e-4
    assert rel(PC.circ_ss_center_point(STEEL, a, t)["k"], 1 / w_ss) < 1e-4


def test_rect_ss_navier_vs_roark_table_case1a():
    mat = STEEL.with_(nu=0.3)
    for r, alpha in ((1.0, 0.0444), (1.4, 0.0770), (2.0, 0.1110), (4.0, 0.1400)):
        got = PR.rect_ss_pressure(mat, r * 0.05, 0.05, 0.002)["alpha"]
        assert rel(got, alpha) < 0.01


def test_rect_ss_point_vs_timoshenko():
    mat = STEEL.with_(nu=0.3); a = 0.05; t = 0.002
    D = PC.flexural_rigidity(mat.E, t, 0.3)
    assert rel(PR.rect_ss_center_point(mat, a, a, t)["k"], D / (0.01160 * a * a)) < 0.01


def test_rect_clamped_vs_timoshenko():
    mat = STEEL.with_(nu=0.3); a = 0.05; t = 0.002
    D = PC.flexural_rigidity(mat.E, t, 0.3)
    assert rel(PR.rect_clamped_pressure(mat, a, a, t)["w_per_q"], 0.00126 * a ** 4 / D) < 0.01
    assert rel(PR.rect_clamped_center_point(mat, a, a, t)["k"], D / (0.00560 * a * a)) < 0.01


def test_large_deflection_small_limit():
    a, t = 0.05, 0.002; w = 1e-6
    q_lin = w / PC.circ_clamped_pressure(STEEL, a, t)["w_per_q"]
    assert rel(PC.circ_clamped_pressure_large(STEEL, a, t, w), q_lin) < 0.002


def test_yield_line_limits():
    mp = 1.0
    assert rel(YL.collapse_rect_pressure(mp, 1, 1), 24) < 1e-12
    assert rel(YL.collapse_rect_pressure(mp, 1e6, 1), 8) < 1e-4
    assert rel(YL.collapse_rect_pressure(mp, 1, 1, clamped=True), 48) < 1e-12
    assert rel(YL.collapse_point_interior(mp), 2 * math.pi) < 1e-12


# ------------------------------------------------------------------ rotating, shells
@pytest.mark.parametrize("Ri", [0.005, 0.02])
def test_annular_disk_vs_bvp(Ri):
    m = STEEL; w = 1200.0; Ro = 0.09
    st_bvp, sr_mid = rotating_disk_bvp(m.rho, m.nu, m.E, w, Ri, Ro)
    assert rel(RD.annular_disk_spin_stress(m.rho, m.nu, w, Ri, Ro), st_bvp) < 1e-5
    assert rel(RD.annular_disk_spin_field(m.rho, m.nu, w, Ri, Ro, 0.5 * (Ri + Ro))[0], sr_mid) < 1e-5


def test_solid_disk_limit():
    m = STEEL; w = 1200.0; R = 0.09
    sc, _ = RD.solid_disk_spin_stress(m.rho, m.nu, w, R)
    sr, st = RD.annular_disk_spin_field(m.rho, m.nu, w, 1e-9, R, 1e-3)
    assert rel(sc, sr) < 1e-3


def test_bar_root_stress_integral():
    rho, w, L = 7850, 1000.0, 0.1
    r = np.linspace(0, L, 100001)
    assert rel(RD.bar_spin_root_stress(rho, w, L), np.trapezoid(rho * w * w * r, r)) < 1e-8


def test_lame_equilibrium_and_bc():
    pi_, ri, ro = 50e6, 0.01, 0.03
    r = np.linspace(ri, ro, 2001)
    sr, st = SH.lame_stresses(pi_, 0.0, ri, ro, r)
    assert abs(sr[0] + pi_) < 1 and abs(sr[-1]) < 1
    res = np.gradient(sr, r) + (sr - st) / r
    assert np.max(np.abs(res[5:-5])) / pi_ * ro < 1e-4


def test_press_fit_vs_lame_displacement():
    E, nu, R, ro = 205e9, 0.29, 0.005, 0.02
    p = SH.press_fit_pressure(5e-6, R, 0.0, ro, E, nu, E, nu)
    # outer radial growth + solid shaft shrink must sum to interference
    sr_o, st_o = SH.lame_stresses(p, 0, R, ro, R)
    u_out = R / E * (st_o - nu * sr_o)
    u_in = R / E * (-p + nu * p)            # solid shaft under external p: sr = st = -p
    assert rel(u_out - u_in, 5e-6) < 1e-9


# ------------------------------------------------------------------ torsion, buckling, joints, vibration
def test_torsion_plastic_ratio():
    d = 0.01; ty = 500e6
    Ty = ty * math.pi * d ** 3 / 16
    assert rel(TS.torsion_plastic_solid(ty, d) / Ty, 4 / 3) < 1e-12


def test_euler_vs_fd_eigen():
    E, I, Lc = 205e9, 1e-10, 0.2
    n = 400; h = Lc / n
    A = (np.diag(np.full(n - 1, 2.0)) - np.diag(np.ones(n - 2), 1) - np.diag(np.ones(n - 2), -1)) / h ** 2
    Pcr = E * I * np.linalg.eigvalsh(A)[0]
    assert rel(BK.euler_load(E, I, Lc), Pcr) < 1e-4


def test_johnson_continuity():
    E, sy, A, I = 205e9, 400e6, 1e-4, 1e-9
    r = math.sqrt(I / A); sl_t = math.sqrt(2 * math.pi ** 2 * E / sy)
    Lt = sl_t * r
    assert rel(BK.johnson_load(E, sy, A, I, Lt * 0.9999), BK.johnson_load(E, sy, A, I, Lt * 1.0001)) < 1e-3


def test_bolt_tensile_area_iso_table():
    assert rel(FJ.bolt_tensile_area(0.004, 0.0007), 8.78e-6) < 0.003
    assert rel(FJ.bolt_tensile_area(0.003, 0.0005), 5.03e-6) < 0.003


@pytest.mark.parametrize("bc,bcs", [("cantilever", {0: [0, 1]}), ("simply_supported", {0: [0], N: [0]}),
                                    ("fixed-fixed", {0: [0, 1], N: [0, 1]}), ("fixed-pinned", {0: [0, 1], N: [0]})])
def test_beam_frequency_vs_fe(bc, bcs):
    mL = STEEL.rho * SEC["A"]
    fe = fe_beam(L, EI, N, bcs, [], mass_per_L=mL)
    assert rel(VM.beam_frequency(STEEL.E, SEC["Ix"], mL, L, bc), fe["f"][0]) < 1e-4


def test_moving_mass_energy():
    d, F = VM.moving_mass_impact(1.3, 3.0, 1e6)
    assert rel(0.5 * 1e6 * d * d, 0.5 * 1.3 * 9.0) < 1e-12


def _virtual_work_membrane(W, H, B, L, s0, support):
    """Independent upper bound: hinges on m + n^2 = 1 with associated flow, power balance P W' = sum(M th' + N d')."""
    M0, N0 = s0 * B * H * H / 4, s0 * B * H
    ext_total = 2 * W / L                        # total hinge extension rate per unit W' (both halves)
    if support == "clamped":
        rot = [1 / L, 1 / L, 2 / L]              # support, support, middle (per unit W')
        ext = [ext_total / 4, ext_total / 4, ext_total / 2]
    else:
        rot = [2 / L]
        ext = [ext_total]
    P = 0.0
    for th, d in zip(rot, ext):
        n = min(d / th * N0 / (2 * M0), 1.0)    # associated flow on m + n^2 = 1
        P += M0 * (1 - n * n) * th + N0 * n * d
    return P


def test_membrane_beam_matches_virtual_work():
    from simblocks.structures.beams.membrane import beam_membrane_point_load
    H, B, L, s0 = 3e-3, 20e-3, 25e-3, 1e9
    for sup in ("clamped", "pinned"):
        for W in np.linspace(0, 3 * H, 31):
            P, Pc, r = beam_membrane_point_load(W, H, B, L, s0, sup)
            assert P == pytest.approx(_virtual_work_membrane(W, H, B, L, s0, sup), rel=1e-9)
        # string asymptote identical for both supports: P = 2 N0 W / L
        P, _, _ = beam_membrane_point_load(3 * H, H, B, L, s0, sup)
        assert P == pytest.approx(2 * s0 * B * H * 3 * H / L)

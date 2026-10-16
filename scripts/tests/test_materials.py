import math, os, tempfile
import numpy as np
import pytest

from simblocks.materials import get, catalog, clt, shore_A_to_E, load_curve, resolve_material, read_curve_csv
from simblocks.materials import curves as CV
from simblocks.materials.base import Isotropic, Lamina

rel = lambda a, b: abs(a / b - 1)
T300 = get("T300_5208")


def test_catalog_sane():
    for k, m in catalog().items():
        if isinstance(m, Isotropic):
            assert 0 < m.nu < 0.5 and m.E > 0 and m.rho > 500 and 0 < m.sy <= m.su * 1.0001, k
        if isinstance(m, Lamina):
            assert m.E1 >= m.E2 and m.nu21 <= m.nu12, k


# ------------------------------------------------------------------ CLT
def _qbar_matrix_method(Q, th):
    """Independent Qbar = T^-1 Q R T R^-1 (Reuter matrix form)."""
    t = math.radians(th); m, n = math.cos(t), math.sin(t)
    T = np.array([[m * m, n * n, 2 * m * n], [n * n, m * m, -2 * m * n], [-m * n, m * n, m * m - n * n]])
    R = np.diag([1, 1, 2.0])
    return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)


@pytest.mark.parametrize("th", [0, 15, 30, 45, 60, 90, -45, 137])
def test_qbar_vs_matrix_transform(th):
    Q = clt.lamina_Q(T300)
    assert np.allclose(clt.lamina_Qbar(Q, th), _qbar_matrix_method(Q, th), rtol=1e-10, atol=1e-3)


def test_parse_layup():
    assert clt.parse_layup("[0/+-45/90]s") == [0, 45, -45, 90, 90, -45, 45, 0]
    assert clt.parse_layup("[0/90]2s") == [0, 90, 0, 90, 90, 0, 90, 0]
    assert clt.parse_layup("[+45/-45]") == [45, -45]
    assert clt.parse_layup("[0_2/90]") == [0, 0, 90]


def test_symmetric_has_no_coupling_and_quasi_iso_is_isotropic():
    abd = clt.laminate_ABD(T300, clt.parse_layup("[0/+-45/90]s"))
    A = abd["A"]
    assert np.abs(abd["B"]).max() < 1e-6 * np.abs(A).max() * abd["h"]
    assert rel(A[0, 0], A[1, 1]) < 1e-10 and abs(A[0, 2]) < 1e-6 * A[0, 0]
    assert rel(A[2, 2], (A[0, 0] - A[0, 1]) / 2) < 1e-10


def test_ud_laminate_matches_lamina():
    abd = clt.laminate_ABD(T300, [0] * 8)
    ec = clt.laminate_engineering_constants(abd)
    assert rel(ec["Ex"], T300.E1) < 1e-10 and rel(ec["Ey"], T300.E2) < 1e-10 and rel(ec["nuxy"], T300.nu12) < 1e-10
    assert rel(ec["Efx"], T300.E1) < 1e-10


def test_unsymmetric_coupling_curvature():
    abd = clt.laminate_ABD(T300, [0, 90])
    e0, k = clt.laminate_response(abd, N=(1000, 0, 0))
    assert abs(k[0]) > 0          # extension-bending coupling present
    assert np.abs(abd["B"]).max() > 0


def test_failure_criteria_uniaxial_limits():
    for s, expect in (((1, 0, 0), T300.Xt), ((-1, 0, 0), T300.Xc), ((0, 1, 0), T300.Yt), ((0, 0, 1), T300.S)):
        for f in (clt.strength_ratio_max_stress, clt.strength_ratio_tsai_hill):
            assert rel(f(s, T300), expect) < 1e-12
    assert rel(clt.strength_ratio_tsai_wu((1, 0, 0), T300), T300.Xt) < 1e-9
    assert rel(clt.strength_ratio_tsai_wu((0, -1, 0), T300), T300.Yc) < 1e-9


def test_first_ply_failure_ud():
    abd = clt.laminate_ABD(T300, [0] * 8)
    R, ply, ang = clt.first_ply_failure(abd, N=(1.0, 0, 0), criterion="max_stress")
    assert rel(R, T300.Xt * abd["h"]) < 1e-9


def test_thermal_free_expansion_ud():
    abd = clt.laminate_ABD(T300, [0] * 4)
    e0, k = clt.laminate_response(abd, dT=100.0)
    assert rel(e0[0], T300.alpha1 * 100) < 1e-6 and rel(e0[1], T300.alpha2 * 100) < 1e-9
    ps = clt.ply_stresses(abd, e0, k, dT=100.0)
    assert max(np.abs(p["top"]).max() for p in ps) < 1e-3    # free expansion -> no stress


def test_halpin_tsai_bounds():
    assert rel(clt.halpin_tsai(230e9, 3.45e9, 0.0, 2), 3.45e9) < 1e-12
    assert rel(clt.halpin_tsai(230e9, 3.45e9, 1.0, 2), 230e9) < 1e-12


def test_laminate_as_isotropic_quasi_iso():
    abd = clt.laminate_ABD(T300, clt.parse_layup("[0/+-45/90]2s"))
    iso = clt.laminate_as_isotropic(abd, use="membrane")
    ec = clt.laminate_engineering_constants(abd)
    assert rel(iso.E, ec["Ex"]) < 1e-9 and 0.2 < iso.nu < 0.4


# ------------------------------------------------------------------ curves
def test_shore_conversion():
    assert 30e6 < shore_A_to_E(95) < 60e6 and shore_A_to_E(20) < shore_A_to_E(60) < shore_A_to_E(90)


def test_ramberg_osgood_passes_through_yield_and_ultimate():
    m = get("7075-T6"); c = CV.ramberg_osgood_curve(m, e_u=0.08)
    e, s = c.eng_strain, c.eng_stress
    assert rel(np.interp(m.sy, s, e) - m.sy / m.E, 0.002) < 3e-3   # grid interpolation
    assert rel(e[-1], 0.08) < 1e-9 and rel(s[-1], m.su) < 1e-12


def test_true_conversion_and_plastic_table_monotone():
    c = CV.ramberg_osgood_curve(get("4130_normalized"))
    et, st = c.true
    assert np.all(np.diff(et) > 0) and np.all(st >= c.eng_stress - 1e-6)
    ep, sp = c.plastic()
    assert ep[0] == 0 and np.all(np.diff(ep) > 0) and np.all(np.diff(sp) >= 0)


def test_csv_roundtrip_with_ksi_and_percent():
    c = CV.ramberg_osgood_curve(get("6061-T6"))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        with open(p, "w") as fh:
            fh.write("# material: test\n# measure: engineering\nstrain_%,stress_ksi\n")
            for a, b in zip(c.eng_strain, c.eng_stress):
                fh.write(f"{a*100},{b/6.894757e6}\n")
        r = read_curve_csv(p, E=c.E)
        assert np.allclose(np.interp(c.eng_strain, r.eng_strain, r.eng_stress), c.eng_stress, rtol=1e-5, atol=1e3)
        c.write_csv(os.path.join(d, "y.csv"))
        r2 = read_curve_csv(os.path.join(d, "y.csv"))
        assert rel(r2.E, c.E) < 1e-5 and "Ramberg" in r2.source


@pytest.mark.parametrize("text,key", [("yea it's 7075 with whatever the best common temper is", "7075-T6"),
                                      ("4130 at 45 HRC", "4130_45HRC"), ("plain 4130 sheet", "4130_normalized"),
                                      ("ti 6al4v", "Ti6Al4V"), ("cast urethane 60A", "urethane_60A"),
                                      ("TPU 85A", "TPU85A_print"), ("lexan", "PC"), ("T300_5208", "T300_5208"),
                                      ("6061", "6061-T6")])
def test_resolver(text, key):
    assert resolve_material(text) == key


def test_jc_kernel_quasistatic_room_temp():
    A, B, n, C, m_, Tm, Tr = get("Ti6Al4V").jc
    assert rel(CV.johnson_cook_stress(0.0, 1.0, Tr, A, B, n, C, m_, Tm, Tr), A) < 1e-12
    assert CV.johnson_cook_stress(0.1, 1e3, Tr, A, B, n, C, m_, Tm, Tr) > CV.johnson_cook_stress(0.1, 1.0, Tr, A, B, n, C, m_, Tm, Tr)


@pytest.mark.parametrize("model,params", [("swift", dict(K_Pa=800e6, eps0=0.01, n=0.2)),
                                          ("voce", dict(sigma0_Pa=300e6, Q_Pa=200e6, b=15)),
                                          ("hollomon", dict(K_Pa=700e6, n=0.18)),
                                          ("ludwik", dict(sigma0_Pa=250e6, K_Pa=400e6, n=0.4)),
                                          ("johnson_cook", dict(A_Pa=350e6, B_Pa=275e6, n=0.36))])
def test_curve_from_fit_reproduces_flow_law(model, params):
    m = get("1006_mild")
    c = CV.curve_from_fit(m, dict(model=model, params=params, citation="test", url="x", location="t1"))
    ep, sp = c.plastic(extend_to=0)
    law = {"swift": lambda e: params["K_Pa"] * (params["eps0"] + e) ** params["n"],
           "voce": lambda e: params["sigma0_Pa"] + params["Q_Pa"] * (1 - np.exp(-params["b"] * e)),
           "hollomon": lambda e: params["K_Pa"] * e ** params["n"],
           "ludwik": lambda e: params["sigma0_Pa"] + params["K_Pa"] * e ** params["n"],
           "johnson_cook": lambda e: params["A_Pa"] + params["B_Pa"] * e ** params["n"]}[model]
    sel = ep > 2e-3
    assert np.allclose(sp[sel], law(ep[sel]), rtol=2e-3)
    assert "test" in c.source


def test_only_exact_condition_fits_become_default(tmp_path):
    from simblocks.materials.curves import load_curve, published_fits
    fits = published_fits()
    for key, recs in fits.items():
        for r in recs:
            assert r.get("applies") in ("exact", "nearest"), key
            assert r.get("citation") and r.get("url") and r.get("location"), key
    c = load_curve("tungsten", curve_dir=str(tmp_path))
    assert c.source.startswith("NO PUBLISHED FIT FOR THIS CONDITION")
    txt = (tmp_path / "WHA_90W.csv").read_text()
    assert "nearest_published_fit" in txt and "NOT THIS CONDITION" in txt


def test_jc_rate_term_applied_to_static_curve(tmp_path):
    from simblocks.materials.curves import load_curve
    import math
    c1 = load_curve("ti64", curve_dir=str(tmp_path), rate=1.0)
    c2 = load_curve("ti64", curve_dir=str(tmp_path), rate=1e-3)
    assert c2.eng_stress[1] == pytest.approx(c1.eng_stress[1])   # never extrapolated below ref
    c3 = load_curve("ti64", curve_dir=str(tmp_path), rate=1e3)
    assert c3.eng_stress[1] / c1.eng_stress[1] == pytest.approx(1 + 0.014 * math.log(1e3), rel=1e-6)

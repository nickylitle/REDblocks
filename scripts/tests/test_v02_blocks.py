"""Tests for v0.2 blocks: fatigue, stress concentration, curved beams/rings, puncture, cards, failure modes, viz."""
import math, os, tempfile
import numpy as np
import pytest

from simblocks.structures import fatigue as FT
from simblocks.structures import stress_concentration as KT
from simblocks.structures import curved as CB
from simblocks.structures import impact as PF
from simblocks.materials import laminate_card, load_card, get, clt
from simblocks.diagnostics.failure import FailureMode, assess_failure_modes, _wilson, candidate_failure_modes

rel = lambda a, b: abs(a / b - 1)


# ------------------------------------------------------------------ fatigue
def test_sn_line_passes_through_anchor_points():
    Sut = 1000e6; Se = 400e6; f = 0.85
    assert rel(FT.sn_life(f * Sut, Sut, Se, f), 1e3) < 1e-9
    assert rel(FT.sn_life(Se * 1.0000001, Sut, Se, f), 1e6) < 1e-4
    assert FT.sn_life(0.9 * Se, Sut, Se, f) == math.inf


def test_mean_stress_criteria_limits():
    Se, Sut, Sy = 400e6, 1000e6, 800e6
    for c in ("goodman", "gerber", "asme_elliptic", "soderberg"):
        assert rel(FT.fatigue_sf(100e6, 0.0, Se, Sut, Sy, c), 4.0) < 1e-9          # no mean stress -> Se / sa
    # Goodman line: on the line SF = 1
    sa = 200e6; sm = (1 - sa / Se) * Sut
    assert rel(FT.fatigue_sf(sa, sm, Se, Sut, criterion="goodman"), 1.0) < 1e-12
    # Gerber: sa/Se + (sm/Sut)^2 = 1 at SF = 1
    sm = Sut * math.sqrt(1 - sa / Se)
    assert rel(FT.fatigue_sf(sa, sm, Se, Sut, criterion="gerber"), 1.0) < 1e-9
    assert FT.fatigue_sf(sa, 300e6, Se, Sut, Sy, "gerber") > FT.fatigue_sf(sa, 300e6, Se, Sut, Sy, "goodman")


def test_marin_and_endurance():
    assert FT.endurance_limit_steel(2000e6) == 700e6 and FT.endurance_limit_steel(600e6) == 300e6
    assert 0.6 < FT.marin_surface_factor(1000e6, "machined") < 0.75
    assert FT.marin_size_factor(0.008) < 1.0 and FT.marin_size_factor(None) == 1.0
    assert rel(FT.miner_damage([1e3, 1e4], [1e4, 1e5]), 0.2) < 1e-12


# ------------------------------------------------------------------ stress concentration
def test_kirsch_peak_and_far_field():
    S = 100e6
    assert rel(KT.kirsch_hoop_stress(S, 1.0, 1.0, math.pi / 2), 3 * S) < 1e-12
    assert rel(KT.kirsch_hoop_stress(S, 1.0, 1.0, 0.0), -S) < 1e-12
    assert rel(KT.kirsch_hoop_stress(S, 1.0, 1e4, math.pi / 2), S) < 1e-6      # far field recovers S (hoop = sigma_x at 90deg)


def test_inglis_circle_and_kf_bounds():
    assert KT.kt_ellipse(1, 1) == 3.0 and KT.kt_ellipse(5, 1) == 11.0
    kf = KT.fatigue_kf(3.0, 0.001, 1000e6)
    assert 1.0 < kf < 3.0
    assert KT.fatigue_kf(3.0, 1.0, 1000e6) > KT.fatigue_kf(3.0, 1e-5, 1000e6)    # bigger radius -> more sensitive


# ------------------------------------------------------------------ curved
def test_winkler_bach_tends_to_straight_beam():
    b, h, M = 0.01, 0.01, 10.0
    si, so, e = CB.curved_beam_rect(M, b, h, r_inner=10.0)
    sb = M * (h / 2) / (b * h ** 3 / 12)
    assert rel(abs(si), sb) < 1e-3 and rel(abs(so), sb) < 1e-3
    si, so, _ = CB.curved_beam_rect(M, b, h, r_inner=0.01)
    assert abs(si) > abs(so)                       # inner fiber governs on tight curvature


def test_ring_matches_roark_table_9_2():
    r = CB.ring_diametral_load(1.0, 1.0, 1.0)
    assert abs(r["dD_load"] - 0.1488) < 1e-4 and abs(r["dD_perp"] - 0.1366) < 1e-4
    assert abs(abs(r["M_load"]) - 1 / math.pi) < 1e-4 and abs(abs(r["M_side"]) - (0.5 - 1 / math.pi)) < 1e-4


# ------------------------------------------------------------------ puncture
def test_shear_plug():
    p = PF.punch_perimeter(w=0.006, l=0.002)
    assert rel(p, 0.016) < 1e-12
    assert rel(PF.shear_plug_capacity(p, 0.002, 1000e6, 0.67), 0.016 * 0.002 * 670e6) < 1e-12
    assert rel(PF.punch_perimeter("circle", d=0.01), math.pi * 0.01) < 1e-12


# ------------------------------------------------------------------ cards
def test_card_ud_matches_lamina_and_roundtrip():
    lam = get("T300_5208")
    c = laminate_card("T300/5208 [0]8")
    assert rel(c.props["Ex"], lam.E1) < 1e-9 and rel(c.props["Sx_fpf"], lam.Xt) < 1e-9
    assert rel(c.props["Sxc_fpf"], lam.Xc) < 1e-9 and rel(c.props["alpha_x"], lam.alpha1) < 1e-6
    with tempfile.TemporaryDirectory() as d:
        c.save(d); c2 = load_card(c.name, d)
        assert c2.props == c.props and c2.angles == c.angles


def test_card_flags_coupling_and_qi():
    qi = laminate_card("T300/5208 [0/+-45/90]2s")
    assert any("quasi-isotropic" in n for n in qi.notes) and not any(n.startswith("UNSYM") for n in qi.notes)
    un = laminate_card("T300/5208 [0/90]")
    assert any(n.startswith("UNSYM") for n in un.notes)
    ub = laminate_card("T300/5208 [30]4")
    assert any(n.startswith("UNBAL") for n in ub.notes)


def test_user_lamina_card_and_isotropic_equivalent():
    c = laminate_card("x", lamina=dict(name="u", E1=100e9, E2=100e9, G12=38.46e9, nu12=0.3, rho=2000,
                                       Xt=500e6, Xc=500e6, Yt=500e6, Yc=500e6, S=289e6, t_ply=0.5e-3), layup="[0]4")
    iso = c.as_isotropic("flexural")
    assert rel(iso.E, 100e9) < 1e-6 and rel(iso.rho, 2000) < 1e-12


# ------------------------------------------------------------------ failure modes
def test_failure_mode_sf_and_assessment():
    fm = [FailureMode("drop", metric="d", limit=5.0), FailureMode("height", metric="z", limit=2.0, fails_when="below")]
    outs = [dict(d=1.0 * i, z=10 - i) for i in range(10)]
    r = assess_failure_modes(fm, dict(d=2.5, z=5), outs, np.arange(10)[:, None] * 1.0, ["x"])
    assert rel(r["drop"]["nominal_SF"], 2.0) < 1e-12 and r["drop"]["n_failed"] == 5
    assert r["height"]["n_failed"] == 2 and rel(r["height"]["nominal_SF"], 2.5) < 1e-12
    assert r["drop"]["governing"] and r["drop"]["drivers"][0][1] < -0.99
    lo, hi = _wilson(0, 400)
    assert lo == 0 and abs(hi - 0.0095) < 3e-4          # rule-of-three-ish upper bound
    assert "puncture by the striking body" in candidate_failure_modes("ramp")


# ------------------------------------------------------------------ viz smoke
def test_viz_outputs_render():
    from simblocks.viz.diagram import Schematic, SignalFlow, glyph_sheet
    with tempfile.TemporaryDirectory() as d:
        s = Schematic("t").body("a", "rotor", "A").body("b", "mass", "B").body("g", "ground", "")
        s.link("a", "b", "contact", "c").link("b", "g", "spring", "k", "1 N/m", "x")
        for p in s.draw(os.path.join(d, "s")) + glyph_sheet(os.path.join(d, "g")):
            assert os.path.getsize(p) > 1000
        f = SignalFlow("t").stage("x", "X", ["a"], "state").stage("y", "Y", [("b", "n")]).edge("x", "y").feedback("y", "x", "l")
        assert all(os.path.getsize(p) > 1000 for p in f.draw(os.path.join(d, "f")))
        cards = laminate_card("T300/5208 [0/90]s").render(os.path.join(d, "c"))
        assert all(os.path.getsize(p) > 1000 for p in cards)


# ------------------------------------------------------------------ layout QA
def test_layout_qa_catches_defects_and_passes_clean():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from simblocks.viz.qa import check_figure
    f, ax = plt.subplots(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "hello world"); ax.text(0.52, 0.5, "overlap here"); ax.plot([0.3, 0.9], [0.51, 0.51])
    ax.text(0.97, 0.1, "clipped text far right side"); ax.text(0.1, 0.9, "tiny", fontsize=4)
    p = " | ".join(check_figure(f))
    assert "OVERLAP" in p and "LINE THROUGH TEXT" in p and "OFF-CANVAS" in p and "TOO SMALL" in p
    g, bx = plt.subplots(); bx.plot([0, 1], [0, 1]); bx.set_title("clean"); bx.set_xlabel("x")
    assert check_figure(g) == []
    plt.close("all")


def test_strict_save_blocks_bad_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from simblocks.viz import theme as T
    from simblocks.viz.qa import LayoutError
    f, ax = plt.subplots(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "aaaa"); ax.text(0.51, 0.5, "bbbb")
    with tempfile.TemporaryDirectory() as d, pytest.raises(LayoutError):
        T.save(f, os.path.join(d, "bad"), strict=True)


@pytest.mark.parametrize("code", ["T300/5208 [0/+-45/90]2s", "Kevlar [0/90]", "IM7 [30/-30]2s", "AS4 [0_2/90]s"])
def test_material_cards_pass_layout_qa(code):
    with tempfile.TemporaryDirectory() as d:
        laminate_card(code).render(os.path.join(d, "c"))      # strict save raises on any layout defect


def test_unreachable_failure_mode_flagged_and_not_governing():
    fm = [FailureMode("drop", metric="d", limit=5.0, reach_metric="dmax"), FailureMode("f", metric="F", limit=10.0)]
    r = assess_failure_modes(fm, dict(d=2.0, dmax=4.0, F=5.0), [dict(d=2.0, dmax=4.0, F=5.0)] * 3)
    assert r["drop"]["reach"]["reachable"] is False and r["f"]["governing"] and not r["drop"]["governing"]

"""Membrane theory blocks: each one against its own exact limits."""
import math

import pytest

from simblocks.structures.plates.large_deflection import (circular_plate_large_deflection, membrane_circular,
                                                          plate_plastic_membrane, plate_strip_large_deflection)

E, NU = 200e9, 0.3


def D(t):
    return E * t ** 3 / (12 * (1 - NU ** 2))


@pytest.mark.parametrize("edge,inplane", [("clamped", "held"), ("clamped", "free"), ("simply_supported", "held"),
                                          ("simply_supported", "free")])
def test_circular_plate_small_load_is_the_linear_plate(edge, inplane):
    a, t, q = 0.1, 2e-3, 1.0
    w_lin = q * a ** 4 / (64 * D(t)) * (1 if edge == "clamped" else (5 + NU) / (1 + NU))
    o = circular_plate_large_deflection(a, t, q, edge=edge, inplane=inplane, E=E, nu=NU)
    assert o["w0"] == pytest.approx(w_lin, rel=1e-6)


def test_circular_point_load_small_is_the_linear_plate():
    a, t, F = 0.1, 2e-3, 1e-3
    o = circular_plate_large_deflection(a, t, F, load="point", E=E, nu=NU)
    assert o["w0"] == pytest.approx(F * a * a / (16 * math.pi * D(t)), rel=1e-6)


def test_held_plate_becomes_hencky_membrane():
    a, t, q = 0.1, 0.2e-3, 2e6          # w0 ~ 50 t: bending is negligible
    p = circular_plate_large_deflection(a, t, q, E=E, nu=NU)
    m = membrane_circular(a, t, q, E=E, nu=NU)
    assert p["w0"] == pytest.approx(m["w0"], rel=0.01)
    assert m["w0"] == pytest.approx(0.6534 * a * (q * a / (E * t)) ** (1 / 3), rel=1e-3)


def test_clamped_plate_matches_published_fe_case():
    # a = 250 mm, t = 5 mm, q = 100 kPa, E = 210 GPa, nu = 0.28: FE 2.326 mm (AutoFEM large-deflection example)
    o = circular_plate_large_deflection(0.25, 0.005, 1e5, E=2.1e11, nu=0.28)
    assert o["w0"] == pytest.approx(2.3258e-3, rel=0.03)


@pytest.mark.parametrize("edge", ["simply_supported", "clamped"])
def test_strip_limits(edge):
    l, t = 0.1, 1e-3
    small = plate_strip_large_deflection(l, t, 1.0, edge=edge, E=E, nu=NU)
    w_lin = (5 if edge == "simply_supported" else 1) * 1.0 * l ** 4 / (384 * D(t))
    assert small["w0"] == pytest.approx(w_lin, rel=1e-3)
    q = 5e8                                  # deep membrane: a cable with S^3 = E t q^2 l^2 / (24 (1 - nu^2))
    big = plate_strip_large_deflection(l, t, q, edge=edge, E=E, nu=NU)
    S = (E * t * q * q * l * l / (24 * (1 - NU ** 2))) ** (1 / 3)
    assert big["S"] == pytest.approx(S, rel=0.05)
    assert big["w0"] == pytest.approx(q * l * l / (8 * S), rel=0.05)


def test_jones_plastic_membrane():
    t, sy = 2e-3, 1000e6
    c = plate_plastic_membrane(t, sy=sy, a=0.05, edge="clamped")
    assert c["Pc"] == pytest.approx(math.pi * sy * t * t)
    assert c["F_tab"][-1] == pytest.approx(c["Pc"] * (1 + c["acc_tab"][-1] / t))
    s = plate_plastic_membrane(t, sy=sy, shape="rectangular", B=0.03, L=0.03, edge="simply_supported")
    assert s["Pc"] == pytest.approx(2 * sy * t * t)            # 8 M0 for a square, simply supported
    assert s["F_tab"][-1] == pytest.approx(s["Pc"] * (1 + 2 * s["acc_tab"][-1] / t))

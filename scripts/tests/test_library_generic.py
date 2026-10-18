"""Generic library blocks: each option returns the same numbers as the specific function it wraps, retired and
merged blocks are hidden from the editor but still importable, and a system file can call a generic block."""
import math

import numpy as np
import pytest

from simblocks.registry import REGISTRY, INTERNAL, is_public
from simblocks.materials import get, LAMINAE
from simblocks.structures.sections.properties import circle, rectangle
from simblocks.structures.beams import static as BS
from simblocks.structures.beams.cases import beam_case, spring_combination
from simblocks.structures.plates import circular as PC, rectangular as PR, yield_line as PY
from simblocks.structures.plates.cases import plate_case, plate_collapse
from simblocks.rigid_body import inertia as IN
from simblocks.rigid_body.spatial import I_box_tensor
from simblocks.structures import rotating as RO
from simblocks.materials import clt
from simblocks.materials import curves as CU
from simblocks.constitutive import rate_damage as RD
from simblocks.thermal import basic as TH

REL = 1e-12
GENERIC = ("beam_case", "spring_combination", "plate_case", "plate_collapse", "mass_properties", "spin_stress",
           "strength_ratio", "hardening_stress", "hyperelastic_stress", "thermal_resistance", "nusselt")

MAT = get("4130_45HRC")
SEC = rectangle(0.02, 0.006)
L = 0.12


def _same(gen, ref, keys=None):
    keys = keys or [k for k in ref if k in gen]
    assert keys
    for k in keys:
        assert gen[k] == pytest.approx(ref[k], rel=REL, abs=0.0), k


def _beam_kw():
    return dict(E=MAT.E, I=SEC["Ix"], Mp=MAT.sy * SEC["Z"], My=MAT.sy * SEC["S"])


# ------------------------------------------------------------------ registry
def test_generic_blocks_public_and_specific_ones_internal():
    for n in GENERIC:
        assert n in REGISTRY and is_public(n), n
    for n in ("cantilever_end_load", "I_disk", "nu_flat_plate", "three_point_normal_loads", "tip_speed",
              "collapse_three_edge_strip", "hollomon_stress", "R_plane_wall"):
        assert n in REGISTRY and not is_public(n), n
    assert all(n in REGISTRY for n in INTERNAL)


def test_generic_docstrings_list_options_and_no_em_dash():
    import inspect
    for n in GENERIC:
        doc = inspect.getdoc(REGISTRY[n].func)
        assert "Returns" in doc and chr(0x2014) not in doc, n


# ------------------------------------------------------------------ beam_case
@pytest.mark.parametrize("support,load,a,ref", [
    ("cantilever", "point", None, lambda: BS.cantilever_end_load(MAT, SEC, L)),
    ("cantilever", "point", 0.07, lambda: BS.cantilever_point_at(MAT, SEC, L, 0.07)),
    ("cantilever", "uniform", None, lambda: BS.cantilever_uniform(MAT, SEC, L)),
    ("simply_supported", "point", None, lambda: BS.simply_supported_center(MAT, SEC, L)),
    ("simply_supported", "point", 0.03, lambda: BS.simply_supported_point_at(MAT, SEC, L, 0.03)),
    ("simply_supported", "uniform", None, lambda: BS.simply_supported_uniform(MAT, SEC, L)),
    ("fixed_fixed", "point", None, lambda: BS.fixed_fixed_center(MAT, SEC, L)),
    ("fixed_fixed", "uniform", None, lambda: BS.fixed_fixed_uniform(MAT, SEC, L)),
    ("propped", "point", None, lambda: BS.propped_center(MAT, SEC, L)),
    ("propped", "uniform", None, lambda: BS.propped_uniform(MAT, SEC, L)),
    ("fixed_guided", "point", None, lambda: BS.fixed_guided_end_load(MAT, SEC, L)),
])
def test_beam_case_matches_specific(support, load, a, ref):
    r = ref()
    g = beam_case(support, load, L, a=a, **_beam_kw())
    _same(g, r, ["k", "delta", "M_max", "x_Mmax", "F_y", "F_c", "Mp"])
    assert g["k"] == r["k"]                                         # stiffness bit-identical
    gEI = beam_case(support, load, L, EI=MAT.E * SEC["Ix"], a=a)
    assert gEI["k"] == r["k"] and "F_c" not in gEI and "F_y" not in gEI


def test_beam_case_cantilever_tip_deflection_and_end_moment():
    g = beam_case("cantilever", "point", L, E=MAT.E, I=SEC["Ix"], a=0.05)
    assert g["delta_tip"] == BS.cantilever_tip_deflection_point_at(1.0, MAT.E, SEC["Ix"], L, 0.05)
    r = BS.cantilever_end_moment(MAT, SEC, L)
    g = beam_case("cantilever", "end_moment", L, **_beam_kw())
    assert g["k"] == pytest.approx(r["k_theta"], rel=REL)
    assert g["tip_delta_per_M"] == pytest.approx(r["tip_delta_per_M"], rel=REL)
    assert g["Mp"] == pytest.approx(r["Mp"], rel=REL) and g["F_y"] == pytest.approx(r["M_y"], rel=REL)


@pytest.mark.parametrize("support,ref", [
    ("cantilever", BS.cantilever_end_load), ("simply_supported", BS.simply_supported_center),
    ("fixed_fixed", BS.fixed_fixed_center), ("fixed_guided", BS.fixed_guided_end_load)])
def test_beam_case_timoshenko(support, ref):
    sec = circle(0.01)
    r = ref(MAT, sec, 0.03, timoshenko=True)
    g = beam_case(support, "point", 0.03, E=MAT.E, I=sec["Ix"], GA=MAT.G * sec["A"] * sec["kappa"])
    assert g["k"] == pytest.approx(r["k"], rel=REL)


@pytest.mark.parametrize("kw,msg", [
    (dict(support="fixed_guided", load="uniform"), "no 'uniform' case"),
    (dict(support="simply_supported", load="end_moment"), "no 'end_moment' case"),
    (dict(support="fixed_fixed", load="point", a=0.03), "centre only"),
    (dict(support="hinged", load="point"), "unknown support"),
    (dict(support="propped", load="uniform", GA=1e6), "shear correction"),
])
def test_beam_case_rejects_missing_cases(kw, msg):
    with pytest.raises(ValueError, match=msg):
        beam_case(L=L, E=MAT.E, I=SEC["Ix"], **kw)


def test_spring_combination():
    assert spring_combination([1e5, 3e5], "series") == BS.springs_in_series(1e5, 3e5)
    assert spring_combination([1e5, 3e5], "parallel") == BS.springs_in_parallel(1e5, 3e5)


# ------------------------------------------------------------------ plate_case / plate_collapse
@pytest.mark.parametrize("args,ref", [
    (("circular", "clamped", "point", 0.04, 0.003), lambda: PC.circ_clamped_center_point(MAT, 0.04, 0.003, 0.0)),
    (("circular", "ss", "point", 0.04, 0.003), lambda: PC.circ_ss_center_point(MAT, 0.04, 0.003, 0.0)),
    (("circular", "clamped", "pressure", 0.04, 0.003), lambda: PC.circ_clamped_pressure(MAT, 0.04, 0.003)),
    (("circular", "ss", "pressure", 0.04, 0.003), lambda: PC.circ_ss_pressure(MAT, 0.04, 0.003)),
    (("rectangular", "ss", "point", 0.10, 0.003), lambda: PR.rect_ss_center_point(MAT, 0.10, 0.06, 0.003, 0.0)),
    (("rectangular", "ss", "pressure", 0.10, 0.003), lambda: PR.rect_ss_pressure(MAT, 0.10, 0.06, 0.003)),
    (("rectangular", "clamped", "point", 0.10, 0.003), lambda: PR.rect_clamped_center_point(MAT, 0.10, 0.06, 0.003)),
    (("rectangular", "clamped", "pressure", 0.10, 0.003), lambda: PR.rect_clamped_pressure(MAT, 0.10, 0.06, 0.003)),
    (("annular", "clamped", "inner_edge", 0.05, 0.003),
     lambda: PC.annular_outer_fixed_inner_load(MAT, 0.05, 0.01, 0.003)),
])
def test_plate_case_matches_specific(args, ref):
    shape = args[0]
    b = 0.01 if shape == "annular" else (0.06 if shape == "rectangular" else None)
    r = ref()
    g = plate_case(*args, b=b, material="4130_45HRC")
    assert g == r
    g2 = plate_case(*args, b=b, E=MAT.E, nu=MAT.nu, sy=MAT.sy)
    assert g2 == r


def test_plate_case_point_at_and_large_deflection():
    r = PR.rect_ss_point_at(MAT, 0.10, 0.06, 0.003, 0.03, 0.02, patch=(0.005, 0.004))
    g = plate_case("rect", "ss", "point_at", 0.10, 0.003, b=0.06, material="4130_45HRC", x0=0.03, y0=0.02,
                   patch_u=0.005, patch_v=0.004)
    assert g == r
    from simblocks.structures.plates.large_deflection import pressure_for_deflection
    q = pressure_for_deflection(0.04, 0.003, 0.002, MAT.E, MAT.nu)      # the exact-limit law (not Roark's 11.8 fit)
    assert plate_case("circular", "clamped", "pressure_large", 0.04, 0.003, material="4130_45HRC", w0=0.002) == dict(q=q)
    assert q == pytest.approx(PC.circ_clamped_pressure_large(MAT, 0.04, 0.003, 0.002), rel=0.02)   # they agree at w0 < t


def test_plate_case_rejects_missing_cases():
    with pytest.raises(ValueError, match="no 'point_at' case"):
        plate_case("circular", "ss", "point_at", 0.04, 0.003, material="4130_45HRC")
    with pytest.raises(ValueError, match="annular"):
        plate_case("annular", "ss", "inner_edge", 0.04, 0.003, b=0.01, material="4130_45HRC")


@pytest.mark.parametrize("case,kw,ref", [
    ("point_interior", dict(), lambda mp: PY.collapse_point_interior(mp, False)),
    ("point_interior", dict(clamped=True), lambda mp: PY.collapse_point_interior(mp, True)),
    ("point_free_edge", dict(), lambda mp: PY.collapse_point_free_edge(mp)),
    ("circular_pressure", dict(a=0.04), lambda mp: PY.collapse_circular_pressure(mp, 0.04, False)),
    ("circular_pressure", dict(a=0.04, clamped=True), lambda mp: PY.collapse_circular_pressure(mp, 0.04, True)),
    ("rect_pressure", dict(a=0.1, b=0.06), lambda mp: PY.collapse_rect_pressure(mp, 0.1, 0.06, False)),
    ("rect_pressure", dict(a=0.1, b=0.06, clamped=True), lambda mp: PY.collapse_rect_pressure(mp, 0.1, 0.06, True)),
    ("three_edge", dict(b_eff=0.05, L_slant=0.03), lambda mp: PY.collapse_three_edge_strip(mp, 0.05, 0.03)),
])
def test_plate_collapse_matches_specific(case, kw, ref):
    mp = PY.plate_mp(MAT.sy, 0.003)
    assert plate_collapse(case, mp=mp, **kw) == dict(capacity=ref(mp), mp=mp)
    assert plate_collapse(case, sy=MAT.sy, t=0.003, **kw)["capacity"] == ref(mp)


# ------------------------------------------------------------------ mass_properties
def _I(shape, **kw):
    from simblocks.rigid_body.inertia import mass_properties
    return mass_properties(shape, **kw)


def test_mass_properties_matches_specific():
    m = 0.4
    g = _I("rod", mass=m, L=0.2)
    assert g["Ixx"] == g["Iyy"] == IN.I_rod_center(m, 0.2) and g["Izz"] == 0.0
    g = _I("disc", mass=m, R=0.05)
    assert g["Izz"] == IN.I_disk(m, 0.05) and g["Ixx"] == IN.I_cylinder_transverse(m, 0.05, 0.0)
    g = _I("ring", mass=m, Ri=0.02, Ro=0.05)
    assert g["Izz"] == IN.I_annulus(m, 0.02, 0.05)
    g = _I("thin_ring", mass=m, R=0.05)
    assert g["Izz"] == IN.I_thin_ring(m, 0.05) and g["Ixx"] == pytest.approx(g["Izz"] / 2, rel=REL)
    g = _I("rect_plate", mass=m, a=0.1, b=0.06)
    assert g["Izz"] == IN.I_rect_plate(m, 0.1, 0.06)
    g = _I("box", mass=m, a=0.1, b=0.06, c=0.03)
    assert np.array_equal(np.array(g["I"]), I_box_tensor(m, 0.1, 0.06, 0.03))
    g = _I("cylinder", mass=m, R=0.03, L=0.1)
    assert g["Izz"] == IN.I_disk(m, 0.03) and g["Ixx"] == g["Iyy"] == IN.I_cylinder_transverse(m, 0.03, 0.1)
    g = _I("tube", mass=m, Ri=0.02, Ro=0.03, L=0.1)
    assert g["Izz"] == IN.I_annulus(m, 0.02, 0.03)
    assert g["Ixx"] == pytest.approx(m * (3 * (0.02 ** 2 + 0.03 ** 2) + 0.1 ** 2) / 12, rel=REL)
    g = _I("sphere", mass=m, R=0.04)
    assert g["Ixx"] == g["Iyy"] == g["Izz"] == IN.I_sphere(m, 0.04)
    assert g["com"] == [0.0, 0.0, 0.0] and g["Ixy"] == g["Ixz"] == g["Iyz"] == 0.0


def test_mass_properties_from_density_and_material():
    rho = MAT.rho
    g = _I("cylinder", material="4130_45HRC", R=0.03, L=0.1)
    assert g["m"] == pytest.approx(rho * math.pi * 0.03 ** 2 * 0.1, rel=REL)
    g = _I("box", density=2700.0, a=0.1, b=0.06, c=0.03)
    assert g["m"] == pytest.approx(2700.0 * 0.1 * 0.06 * 0.03, rel=REL)
    with pytest.raises(ValueError, match="needs 't'"):
        _I("disc", density=2700.0, R=0.05)
    with pytest.raises(ValueError, match="unknown shape"):
        _I("cone", mass=1.0, R=0.05)


# ------------------------------------------------------------------ spin_stress
def test_spin_stress_matches_specific():
    rho, nu, w = 7850.0, 0.29, 900.0
    sc, st = RO.solid_disk_spin_stress(rho, nu, w, 0.1)
    g = RO.spin_stress("solid_disk", omega=w, rho=rho, nu=nu, R=0.1, su=1.5e9)
    assert (g["sigma_max"], g["sigma_center"], g["sigma_t_rim"]) == (sc, sc, st)
    assert g["burst_rpm"] == RO.burst_rpm_solid_disk(rho, nu, 1.5e9, 0.1)
    g = RO.spin_stress("annular_disk", omega=w, rho=rho, nu=nu, Ri=0.02, Ro=0.1, r=0.05)
    assert g["sigma_max"] == RO.annular_disk_spin_stress(rho, nu, w, 0.02, 0.1)
    assert (g["sr"], g["st"]) == RO.annular_disk_spin_field(rho, nu, w, 0.02, 0.1, 0.05)
    assert RO.spin_stress("thin_ring", omega=w, rho=rho, R=0.1)["sigma_max"] == RO.thin_ring_spin_stress(rho, w, 0.1)
    assert RO.spin_stress("bar_root", omega=w, rho=rho, L=0.1, A=1e-4, m_tip=0.02, r_hub=0.01)["sigma_max"] == \
        RO.bar_spin_root_stress(rho, w, 0.1, 1e-4, 0.02, 0.01)
    rpm = 9000.0
    assert RO.spin_stress("thin_ring", rpm=rpm, rho=rho, R=0.1)["sigma_max"] == \
        RO.thin_ring_spin_stress(rho, rpm * 2 * math.pi / 60, 0.1)


def test_spin_stress_burst_scaling_and_material():
    g = RO.spin_stress("thin_ring", omega=500.0, material="4130_45HRC", R=0.1)
    w_b = g["burst_rpm"] * 2 * math.pi / 60
    assert RO.thin_ring_spin_stress(MAT.rho, w_b, 0.1) == pytest.approx(MAT.su, rel=1e-9)
    assert g["sigma_max"] == RO.thin_ring_spin_stress(MAT.rho, 500.0, 0.1)
    assert set(RO.spin_stress("solid_disk", rho=7850.0, nu=0.3, su=1e9, R=0.1)) == {"burst_rpm"}


# ------------------------------------------------------------------ strength_ratio
@pytest.mark.parametrize("crit,ref", [("max_stress", clt.strength_ratio_max_stress),
                                      ("tsai_hill", clt.strength_ratio_tsai_hill),
                                      ("tsai_wu", clt.strength_ratio_tsai_wu)])
@pytest.mark.parametrize("s", [(800e6, 20e6, 30e6), (-600e6, -90e6, -40e6), (0.0, 30e6, 0.0)])
def test_strength_ratio_matches_specific(crit, ref, s):
    lam = LAMINAE["T300_5208"]
    r = ref(s, lam)
    assert clt.strength_ratio(crit, *s, lamina="T300_5208") == r
    assert clt.strength_ratio(crit, *s, Xt=lam.Xt, Xc=lam.Xc, Yt=lam.Yt, Yc=lam.Yc, S=lam.S) == r


# ------------------------------------------------------------------ hardening_stress / hyperelastic_stress
def _py(f):
    return getattr(f, "py_func", f)


@pytest.mark.parametrize("eps", [0.0, 0.01, 0.2])
def test_hardening_stress_matches_specific(eps):
    assert CU.hardening_stress("hollomon", eps, K=900e6, n=0.2) == _py(CU.hollomon_stress)(eps, 900e6, 0.2)
    assert CU.hardening_stress("swift", eps, K=900e6, eps0=0.002, n=0.2) == \
        _py(CU.swift_stress)(eps, 900e6, 0.002, 0.2)
    assert CU.hardening_stress("voce", eps, sigma0=300e6, Q=200e6, b=12.0) == \
        _py(CU.voce_stress)(eps, 300e6, 200e6, 12.0)
    assert CU.hardening_stress("ludwik", eps, sigma0=300e6, K=500e6, n=0.4) == \
        _py(CU.ludwik_stress)(eps, 300e6, 500e6, 0.4)
    jc = dict(A=792e6, B=510e6, n=0.26, C=0.014, m=1.03, Tm=1793.0, Tr=293.0)
    assert CU.hardening_stress("johnson_cook", eps, rate=1e3, T=500.0, **jc) == \
        _py(CU.johnson_cook_stress)(eps, 1e3, 500.0, 792e6, 510e6, 0.26, 0.014, 1.03, 1793.0, 293.0, 1.0)
    with pytest.raises(ValueError, match="needs"):
        CU.hardening_stress("swift", eps, K=900e6, n=0.2)


@pytest.mark.parametrize("lam", [0.7, 1.0, 1.8])
def test_hyperelastic_stress_matches_specific(lam):
    assert RD.hyperelastic_stress("neo_hookean", lam, mu=1e6) == _py(RD.neo_hookean_uniaxial)(lam, 1e6)
    assert RD.hyperelastic_stress("mooney_rivlin", lam, C10=0.4e6, C01=0.1e6) == \
        _py(RD.mooney_rivlin_uniaxial)(lam, 0.4e6, 0.1e6)


# ------------------------------------------------------------------ thermal_resistance / nusselt
def test_thermal_resistance_matches_specific():
    assert TH.thermal_resistance("plane_wall", L=0.003, k=200.0, A=1e-3) == TH.R_plane_wall(0.003, 200.0, 1e-3)
    assert TH.thermal_resistance("cylinder", ri=0.01, ro=0.012, k=50.0, L=0.04) == TH.R_cylinder(0.01, 0.012, 50.0, 0.04)
    assert TH.thermal_resistance("sphere", ri=0.01, ro=0.02, k=1.0) == TH.R_sphere(0.01, 0.02, 1.0)
    assert TH.thermal_resistance("convection", h=25.0, A=0.01) == TH.R_convection(25.0, 0.01)
    with pytest.raises(ValueError, match="needs"):
        TH.thermal_resistance("cylinder", ri=0.01, k=50.0, L=0.04)


@pytest.mark.parametrize("Re", [2e4, 8e5])
def test_nusselt_matches_specific(Re):
    Pr = 0.71
    assert TH.nusselt("flat_plate", Pr, Re=Re) == TH.nu_flat_plate(Re, Pr)
    assert TH.nusselt("cylinder_crossflow", Pr, Re=Re) == TH.nu_cylinder_crossflow(Re, Pr)
    assert TH.nusselt("pipe_gnielinski", Pr, Re=Re) == TH.nu_pipe_gnielinski(Re, Pr)
    assert TH.nusselt("vertical_plate_natural", Pr, Ra=Re * 1e3) == TH.nu_vertical_plate_natural(Re * 1e3, Pr)
    assert TH.nusselt("horizontal_cylinder_natural", Pr, Ra=Re) == TH.nu_horizontal_cylinder_natural(Re, Pr)
    with pytest.raises(ValueError, match="needs Ra"):
        TH.nusselt("vertical_plate_natural", Pr, Re=Re)


# ------------------------------------------------------------------ system file
def test_system_file_calls_generic_blocks_from_derived():
    from simblocks.compiler.system import load_system
    s = load_system("""simblocks: 1
name: generic blocks
dim: 1d
params:
  E: 205 GPa
  I: 360 mm^4
  L: 120 mm
derived:
  arm:  {block: beam_case, support: cantilever, load: point, L: $L, E: $E, I: $I}
  lip:  {block: plate_case, shape: circular, edges: clamped, load: point, a: 40 mm, t: 3 mm, material: 4130_45HRC}
  rotor: {block: mass_properties, shape: disc, mass: 0.2 kg, R: 50 mm}
  k_both: {block: spring_combination, k: [$arm.k, $lip.k], arrangement: series}
bodies:
  tip: {dof: translation, mass: 0.1 kg}
elements:
  blade: {type: spring, measure: tip, k: $k_both}
outputs:
  x: {body: tip, quantity: position, reduce: max}
run: {t_end: 1 ms, dt: auto}
""")
    ctx = s.nominal.ctx
    E, I, L_ = 205e9, 360e-12, 0.12
    assert ctx["arm"]["k"] == pytest.approx(3 * E * I / L_ ** 3, rel=1e-12)
    assert ctx["lip"] == PC.circ_clamped_center_point(MAT, 0.04, 0.003, 0.0)
    assert ctx["rotor"]["Izz"] == pytest.approx(IN.I_disk(0.2, 0.05), rel=1e-12)
    assert ctx["k_both"] == pytest.approx(BS.springs_in_series(ctx["arm"]["k"], ctx["lip"]["k"]), rel=1e-12)

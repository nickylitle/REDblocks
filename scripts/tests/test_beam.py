"""Co-rotational beams against closed-form answers."""
import math

import numpy as np
import pytest

from simblocks.compiler.system import load_system

E, b, h, rho, L = 200e9, 0.02, 0.004, 7850.0, 0.3
I = b * h ** 3 / 12; A = b * h; m = rho * A * L


def cant(dim, extra_el="", extra_out="", n=10, t_end=0.6, tipmass=1e-4, damping=0.0, load=None, v0=None):
    v0 = v0 or ("[0, 0]" if dim == "2d" else "[0, 0, 0]")
    two = dim == "2d"
    pt = "[0, 0]" if two else "[0, 0, 0]"
    at = f"[{L}, 0]" if two else f"[{L}, 0, 0]"
    ld = ""
    if load is not None:   # a tip damper near critical for the first mode brings it to rest
        ld = f"  push: {{type: force, P: {load}, measure: tip.y}}\n  calm: {{type: dashpot, c: 30, measure: tip.y}}\n"
    body = (f"  tip: {{dof: planar, mass: {tipmass} kg, inertia: {tipmass * 1e-4}, at: {at}, v0: {v0}, points: {{p: {pt}}}}}" if two else
            f"  tip: {{dof: free, mass: {tipmass} kg, inertia: {tipmass * 1e-4}, at: {at}, v0: {v0}, points: {{p: {pt}}}}}")
    sec = "{block: rectangle, b: 20 mm, h: 4 mm}"
    return load_system(f"""simblocks: 1
name: t
dim: {dim}
params:
  sec: {sec}
bodies:
  wall: {{ground: true, points: {{root: {pt}}}}}
{body}
elements:
  bar: {{type: beam, E: 200 GPa, density: 7850 kg/m^3, section: $sec, segments: {n},
         damping: {damping}, measure: [wall.root, tip.p]{extra_el}}}
{ld}outputs:
  ymin: {{body: tip, quantity: y, reduce: min}}
  yfin: {{body: tip, quantity: y}}
  M: {{element: bar, quantity: moment, reduce: max_abs}}
{extra_out}run: {{t_end: {t_end} s}}
""")


def test_cantilever_static_tip_load_2d():
    """Tip load P, heavily damped to rest: tip deflection P L^3 / 3 E I, root moment P L."""
    P = -20.0
    S = cant("2d", n=8, t_end=1.5, tipmass=0.05, damping=0.3, load=P)
    o = S.run()
    assert o["yfin"] == pytest.approx(P * L ** 3 / (3 * E * I), rel=0.01)
    assert o["M"] >= abs(P) * L * 0.99


def test_cantilever_first_frequency_2d():
    """Free vibration from a small tip kick, light tip: first mode 1.875^2 sqrt(EI / (rho A L^4))."""
    S = cant("2d", n=12, t_end=0.5, tipmass=1e-4, v0="[0, 0.05]")
    o, H = S.run(history=True, n_log=20000)
    t, y = H["t"], H["tip.y"] - H["tip.y"].mean()
    N = len(y) * 16
    f = np.fft.rfftfreq(N, t[1] - t[0]); X = np.abs(np.fft.rfft(y * np.hanning(len(y)), N))
    w = 2 * np.pi * f[np.argmax(X)]
    w1 = 1.875104 ** 2 * math.sqrt(E * I / (rho * A * L ** 4))
    assert w == pytest.approx(w1, rel=0.02)


def test_large_rotation_beam_whirls_without_stretching():
    """A beam spun about its root by a fast-turning hub: centrifugal stiffening only, no spurious stretch from
    the rotation itself (the co-rotational frame removes rigid rotation)."""
    S = load_system("""simblocks: 1
name: t
dim: 2d
params:
  sec: {block: rectangle, b: 20 mm, h: 4 mm}
bodies:
  hub: {dof: [rz], inertia: 10 kg m^2, omega0: 20 rad/s, points: {root: [0, 0]}}
  tip: {mass: 0.001 kg, inertia: 1e-9, at: [0.3, 0], v0: [0, 6], omega0: 20 rad/s, points: {p: [0, 0]}}
elements:
  bar: {type: beam, E: 200 GPa, density: 7850 kg/m^3, section: $sec, segments: 6, measure: [hub.root, tip.p]}
outputs:
  st: {element: bar, quantity: measure, reduce: max_abs}
  M: {element: bar, quantity: moment, reduce: max_abs}
run: {t_end: 0.5 s}
""")
    o = S.run()
    # centrifugal stretch of a whirling bar: ~ rho w^2 L^3 / (3 E) (tiny); bending stays near zero
    assert o["st"] < 1e-5 and o["M"] < 0.5
    assert o["_ledger_rel"] < 0.01


def test_plastic_hinge_collapse_2d():
    """Elastic-perfectly-plastic cantilever (hinge: auto, Mp = sy Z): a tip load above Mp/L keeps bending."""
    from simblocks.materials import get
    sy = get("4130_normalized").sy; Z = b * h * h / 4; Mp = sy * Z
    S = cant("2d", n=6, t_end=0.6, tipmass=0.05, damping=0.2, load=-1.2 * Mp / L,
             extra_el=", hinge: auto, material: 4130_normalized", extra_out="  hs: {element: bar, quantity: hinge_set, reduce: max}\n")
    o = S.run()
    assert o["hs"] > 0.05                                    # a real hinge has formed and keeps turning
    S2 = cant("2d", n=6, t_end=0.6, tipmass=0.05, damping=0.2, load=-0.7 * Mp / L,
              extra_el=", hinge: auto, material: 4130_normalized", extra_out="  hs: {element: bar, quantity: hinge_set, reduce: max}\n")
    assert S2.run()["hs"] < 1e-3                              # below collapse (dynamic overshoot damped)


# ---------------------------------------------------------------- 3D
def cant3(load_meas, P, orient="y", n=6, t_end=1.5, extra_el="", extra="", extra_out="", damper=None):
    damper = damper or f"tip.{load_meas.split('.')[-1]}"
    return load_system(f"""simblocks: 1
name: t
dim: 3d
bodies:
  wall: {{ground: true, points: {{root: [0, 0, 0]}}}}
  tip: {{dof: free, mass: 0.05 kg, inertia: 2e-6, at: [{L}, 0, 0], points: {{p: [0, 0, 0]}}}}
elements:
  bar: {{type: beam, E: 200 GPa, density: 7850 kg/m^3, section: {{block: rectangle, b: 20 mm, h: 4 mm}},
         segments: {n}, damping: 0.3, orient: {orient}, measure: [wall.root, tip.p]{extra_el}}}
  push: {{type: force, P: {P}, measure: {load_meas}}}
  calm: {{type: dashpot, c: 30, measure: {damper}}}
{extra}outputs:
  y: {{body: tip, quantity: y}}
  z: {{body: tip, quantity: z}}
  M: {{element: bar, quantity: moment}}
{extra_out}run: {{t_end: {t_end} s}}
""")


@pytest.mark.parametrize("axis, Iuse", [("y", b * h ** 3 / 12), ("z", h * b ** 3 / 12)])
def test_cantilever_static_3d_both_planes(axis, Iuse):
    """orient: y puts the section's h along y: a y load bends it the weak way (b h^3 / 12), a z load the strong way."""
    P = -20.0
    o = cant3(f"tip.{axis}", P).run()
    assert o[axis] == pytest.approx(P * L ** 3 / (3 * E * Iuse), rel=0.01)
    assert o["M"] == pytest.approx(abs(P) * L, rel=0.01)
    assert o["_ledger_rel"] < 1e-3


def test_torsion_3d():
    """Tip torque T on a beam: twist T L / G J, internal torque T."""
    S = load_system(f"""simblocks: 1
name: t
dim: 3d
bodies:
  wall: {{ground: true, points: {{root: [0, 0, 0]}}}}
  tip: {{dof: free, mass: 0.05 kg, inertia: 2e-6, at: [{L}, 0, 0], points: {{p: [0, 0, 0]}}}}
elements:
  bar: {{type: beam, EA: 1.6e7, EI: 2000, EIy: 2000, GJ: 100, mass: 0.2, segments: 4, damping: 0.3,
         measure: [wall.root, tip.p]}}
  twist: {{type: force, P: 2.0, measure: {{twist: [wall, tip], axis: x}}}}
  calm: {{type: dashpot, c: 0.02, measure: {{twist: [wall, tip], axis: x}}}}
outputs:
  T: {{element: bar, quantity: torque}}
  ang: {{element: twist, quantity: measure}}
run: {{t_end: 1.0 s}}
""")
    o = S.run()
    assert o["ang"] == pytest.approx(2.0 * L / 100.0, rel=1e-3)
    assert abs(o["T"]) == pytest.approx(2.0, rel=1e-3)


def test_beam_rotated_in_space_matches_axis_aligned():
    """The same cantilever built along a skew direction (load square to it) deflects the same amount."""
    d = np.array([1.0, 2.0, 2.0]) / 3.0; nrm = np.array([2.0, 1.0, -2.0]) / 3.0     # orthonormal pair
    tip = [float(x) for x in d * L]; nrm = [float(x) for x in nrm]; P = -20.0
    S = load_system(f"""simblocks: 1
name: t
dim: 3d
bodies:
  wall: {{ground: true, points: {{root: [0, 0, 0]}}}}
  tip: {{dof: free, mass: 0.05 kg, inertia: 2e-6, at: {tip}, points: {{p: [0, 0, 0]}}}}
  ref: {{ground: true, points: {{o: [0, 0, 0]}}}}
elements:
  bar: {{type: beam, E: 200 GPa, density: 7850 kg/m^3, section: {{block: rectangle, b: 20 mm, h: 4 mm}},
         segments: 6, damping: 0.3, orient: {nrm}, measure: [wall.root, tip.p]}}
  push: {{type: force, P: {P}, measure: {{along: [ref.o, tip.p], dir: {nrm}}}}}
  calm: {{type: dashpot, c: 30, measure: {{along: [ref.o, tip.p], dir: {nrm}}}}}
outputs:
  defl: {{element: push, quantity: measure}}
run: {{t_end: 1.5 s}}
""")
    o = S.run()
    assert o["defl"] == pytest.approx(P * L ** 3 / (3 * E * b * h ** 3 / 12), rel=0.01)


def test_plastic_hinge_3d():
    """hinge: auto in 3D: above Mp / L the tip keeps going, below it no set."""
    from simblocks.materials import get
    sy = get("4130_normalized").sy; Mp = sy * b * h * h / 4
    out = "  hs: {element: bar, quantity: hinge_set, reduce: max}\n"
    hi = cant3("tip.y", -1.2 * Mp / L, n=4, t_end=0.6, extra_el=", hinge: auto, material: 4130_normalized",
               extra_out=out).run()
    lo = cant3("tip.y", -0.7 * Mp / L, n=4, t_end=0.6, extra_el=", hinge: auto, material: 4130_normalized",
               extra_out=out).run()
    assert hi["hs"] > 0.05 and lo["hs"] < 1e-3


# ---- the general cantilever: every load case against Roark Table 8.1
def test_general_cantilever_cases():
    import os, tempfile
    from simblocks.structures.beams.cantilever_general import cantilever
    EI, L = 100.0, 1.0
    assert cantilever(L, EI=EI, value=10)["tip_deflection"] == pytest.approx(10 / (3 * EI), rel=1e-5)
    assert cantilever(L, EI=EI, load="moment", value=5)["tip_deflection"] == pytest.approx(5 / (2 * EI), rel=1e-5)
    assert cantilever(L, EI=EI, load="uniform", value=12)["tip_deflection"] == pytest.approx(12 / (8 * EI), rel=1e-5)
    a = 0.5
    r = cantilever(L, EI=EI, load="uniform", value=12, a=a, b=L)
    assert r["tip_deflection"] == pytest.approx(12 / (24 * EI) * (3 * L ** 4 - 4 * a ** 3 * L + a ** 4), rel=1e-3)
    r = cantilever(L, EI=EI, load="linear", value=0, value_b=12)
    assert r["tip_deflection"] == pytest.approx(11 * 12 * L ** 4 / (120 * EI), rel=1e-4)
    assert r["root_moment"] == pytest.approx(4.0, rel=1e-4)
    r = cantilever(L, EI=EI, load="point", value=10, a=0.4)
    assert r["tip_deflection"] == pytest.approx(10 * 0.16 * (3 - 0.4) / (6 * EI), rel=1e-4)
    r = cantilever(L, EI=EI, value=10, more_loads=[{"load": "uniform", "value": 12}])
    assert r["tip_deflection"] == pytest.approx(10 / (3 * EI) + 12 / (8 * EI), rel=1e-5)
    d = tempfile.mkdtemp(); f = os.path.join(d, "w.csv")
    open(f, "w").write("x_mm,w\n0,12\n1000,12\n")
    r = cantilever(L, EI=EI, load="csv", csv_file=f, csv_x_unit="mm")
    assert r["tip_deflection"] == pytest.approx(12 / (8 * EI), rel=1e-4)


def test_general_cantilever_stress_from_material_and_section():
    from simblocks.structures.beams.cantilever_general import cantilever
    from simblocks.structures.sections.properties import rectangle
    sec = rectangle(0.02, 0.004)
    r = cantilever(0.3, mat="6061-T6", sec=sec, value=10)
    assert r["max_stress"] == pytest.approx(10 * 0.3 * 0.002 / sec["Ix"], rel=1e-6)

"""Force elements driven by a CSV table (against time or against the measure)."""
import math

import numpy as np
import pytest

from simblocks.compiler.system import load_system
from simblocks.compiler.loader import SpecError


def model(tmp_path, csvtext, opts, dim="1d", t_end="40 ms", name="load.csv"):
    (tmp_path / name).write_text(csvtext)
    if dim == "1d":
        bodies = "  b: {dof: translation, mass: 1 kg}"; meas = "b"; out = "b, quantity: velocity"
    elif dim == "2d":
        bodies = "  g: {ground: true, points: {o: [0, 0]}}\n  b: {dof: planar, mass: 1 kg, inertia: 1e-3, points: {p: [0, 0]}}"
        meas = "b.y"; out = "b, quantity: vy"
    else:
        bodies = "  g: {ground: true, points: {o: [0, 0, 0]}}\n  b: {dof: free, mass: 1 kg, inertia: 1e-3, points: {p: [0, 0, 0]}}"
        meas = "b.z"; out = "b, quantity: vz"
    p = tmp_path / "m.yaml"
    p.write_text(f"""simblocks: 1
name: t
dim: {dim}
bodies:
{bodies}
elements:
  push: {{type: force, measure: {meas}, csv: {{file: {name}, {opts}}}}}
outputs:
  v: {{body: {out}}}
run: {{t_end: {t_end}}}
""")
    return load_system(str(p))


RAMP = "time_ms,load_kN\n0,0\n10,1\n20,1\n30,0\n"


@pytest.mark.parametrize("dim", ["1d", "2d", "3d"])
def test_time_table_impulse(tmp_path, dim):
    """Trapezoid pulse 0 -> 1 kN over 10 ms, held 10 ms, back to 0: impulse 20 N s on 1 kg."""
    o = model(tmp_path, RAMP, "x_unit: ms, y_unit: kN", dim).run()
    assert o["v"] == pytest.approx(20.0, rel=1e-6)
    assert o["_ledger_rel"] < 1e-9


def test_step_interpolation(tmp_path):
    """step holds each value to the next row: impulse = 1 kN x 20 ms (rows at 10 and 20 ms hold 1 kN)."""
    o = model(tmp_path, RAMP, "x_unit: ms, y_unit: kN, interpolation: step").run()
    assert o["v"] == pytest.approx(20.0, rel=1e-3)
    o2 = model(tmp_path, "t,F\n0,0\n10,1\n30,0\n", "x_unit: ms, y_unit: kN, interpolation: step").run()
    assert o2["v"] == pytest.approx(20.0, rel=1e-3)       # 0 for 10 ms, 1 kN for 20 ms, then 0


def test_smooth_interpolation_no_overshoot(tmp_path):
    """PCHIP passes through the points and never overshoots a flat step."""
    from simblocks.compiler.loadtable import _pchip
    x = np.array([0.0, 1.0, 2.0, 3.0]); y = np.array([0.0, 0.0, 1.0, 1.0])
    xs, ys = _pchip(x, y)
    assert ys.min() >= -1e-12 and ys.max() <= 1 + 1e-12
    for xi, yi in zip(x, y):
        assert np.interp(xi, xs, ys) == pytest.approx(yi)
    o = model(tmp_path, RAMP, "x_unit: ms, y_unit: kN, interpolation: smooth").run()
    # PCHIP end slopes are 1.5x the end secants (as SciPy), so the ramps bulge: area 22.5 N s, not 20
    assert o["v"] == pytest.approx(22.5, rel=1e-3)


def test_columns_by_name_and_held_ends(tmp_path):
    """Pick columns by header; before the first row and after the last the end values hold."""
    txt = "# rig data\nidx;t_s;F_N\n1;0.01;5\n2;0.02;5\n"
    o = model(tmp_path, txt, "x_col: t_s, y_col: F_N", t_end="40 ms").run()
    assert o["v"] == pytest.approx(5 * 0.04, rel=1e-6)    # 5 N held the whole 40 ms


def test_against_measure_is_a_nonlinear_spring(tmp_path):
    """F(s) table against the measure behaves as an elastic curve: energy is conserved through a bounce."""
    (tmp_path / "k.csv").write_text("s_mm,F_N\n-10,1000\n0,0\n10,-1000\n")
    p = tmp_path / "m.yaml"
    p.write_text("""simblocks: 1
name: t
dim: 1d
bodies:
  b: {dof: translation, mass: 1 kg, v0: 1 m/s}
elements:
  spr: {type: force, measure: b, csv: {file: k.csv, against: measure, x_unit: mm}}
outputs:
  x: {body: b, quantity: position, reduce: max}
run: {t_end: 0.2 s}
""")
    o = load_system(str(p)).run()
    k = 1000 / 0.01
    assert o["x"] == pytest.approx(1.0 / math.sqrt(k / 1.0), rel=1e-3)   # linear table = spring k: amplitude v/w


@pytest.mark.parametrize("opts, msg", [
    ("x_unit: mm, y_unit: N", "not a time"),
    ("x_unit: ms, y_unit: m", "not a force"),
    ("x_unit: ms, interpolation: cubic", "linear, step or smooth"),
    ("x_unit: ms, x_col: nope", "header"),
])
def test_bad_options_refused(tmp_path, opts, msg):
    with pytest.raises(SpecError, match=msg):
        model(tmp_path, RAMP, opts)


def test_unsorted_and_missing_file(tmp_path):
    with pytest.raises(SpecError, match="must increase"):
        model(tmp_path, "t,F\n0,1\n2,1\n1,0\n", "x_unit: s")
    p = tmp_path / "m.yaml"
    p.write_text("""simblocks: 1
name: t
dim: 1d
bodies:
  b: {dof: translation, mass: 1 kg}
elements:
  push: {type: force, measure: b, csv: {file: nothere.csv}}
outputs:
  v: {body: b, quantity: velocity}
run: {t_end: 1 s}
""")
    with pytest.raises(SpecError, match="cannot open"):
        load_system(str(p))

"""Model blocks: a model with ports used as a block in another; values and time series pass in memory, upstream
first; from_csv and export headers read straight back in."""
import pytest

from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError

HIT = """simblocks: 1
name: hit
dim: 1d
ports:
  inputs:
    v_in: {unit: m/s, default: 2 m/s}
  outputs:
    v_out: {output: v}
    F_out: {element: s, quantity: force, unit: N}
bodies:
  a: {dof: translation, mass: 1 kg, v0: $v_in}
elements:
  s: {type: kelvin_voigt, measure: a, one_sided: true, k: 1e5, c: 10}
outputs:
  F: {element: s, quantity: force, reduce: max, unit: kN}
  v: {body: a, quantity: velocity, unit: mm/s}
exports:
  - name: loads
    columns: [{element: s, quantity: force, unit: kN, label: Fz}, {body: a, quantity: velocity, unit: m/s, label: va}]
    time_unit: ms
    points: all
run: {t_end: 30 ms}
"""
FRAME = """simblocks: 1
name: frame
dim: 1d
ports:
  inputs:
    F_hit: {kind: series, unit: N}
    v0: {unit: m/s}
  outputs:
    v_end: {output: vb}
bodies:
  b: {dof: translation, mass: 2 kg, v0: $v0}
elements:
  push: {type: force, measure: b, csv: {series: $F_hit}}
outputs:
  vb: {body: b, quantity: velocity}
run: {t_end: 30 ms}
"""
SEQ = """simblocks: 1
name: seq
params: {v: 3 m/s}
models:
  hit: {file: hit.yaml, inputs: {v_in: $v}}
  frame: {file: frame.yaml, inputs: {F_hit: $hit.F_out, v0: $hit.v_out}}
outputs:
  twice: {expr: "$frame.v_end * 2"}
"""


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "hit.yaml").write_text(HIT)
    (tmp_path / "frame.yaml").write_text(FRAME)
    (tmp_path / "seq.yaml").write_text(SEQ)
    return tmp_path


def test_sequence_passes_values_and_series(folder):
    o = load_system(str(folder / "seq.yaml")).run()
    v_out = o["hit.v_out"]
    assert -3.0 < v_out < 0.0                                   # the hit bounces back, losing a little to damping
    # the frame takes the hit's whole impulse on 2 kg, starting at the hit's rebound speed
    assert o["frame.v_end"] == pytest.approx(v_out + (3.0 - v_out) / 2.0, rel=2e-4)
    assert o["twice"] == pytest.approx(2 * o["frame.v_end"])
    assert o["hit.F"] > 0                                       # every block's outputs are listed


def test_inputs_change_reruns_the_block(folder):
    S = load_system(str(folder / "seq.yaml"))
    a = S.run(); b = S.run({"v": "1 m/s"})
    assert b["hit.v_out"] == pytest.approx(a["hit.v_out"] / 3.0, rel=1e-6)   # linear contact: speed scales


def test_port_without_default_runs_only_as_a_block(folder):
    with pytest.raises(SpecError) as ei:
        load_system(str(folder / "frame.yaml"))
    assert "runs only as a block" in str(ei.value)
    assert load_system(str(folder / "hit.yaml")).run()["v"] < 0       # the default lets it run on its own


def test_loop_is_an_error(folder):
    (folder / "a.yaml").write_text(SEQ.replace("inputs: {v_in: $v}", "inputs: {v_in: $frame.v_end}"))
    with pytest.raises(SpecError) as ei:
        load_system(str(folder / "a.yaml"))
    assert "loop" in str(ei.value)


def test_unknown_port_is_named(folder):
    (folder / "b.yaml").write_text(SEQ.replace("v_in: $v", "speed: $v"))
    with pytest.raises(SpecError) as ei:
        load_system(str(folder / "b.yaml"))
    assert "no input port 'speed'" in str(ei.value)


def test_export_reads_back_by_header(folder):
    """An export's CSV feeds a force table (units from its header) and from_csv values."""
    S = load_system(str(folder / "hit.yaml")); S.run_with_exports()
    (folder / "next.yaml").write_text("""simblocks: 1
name: next
dim: 1d
params:
  v_last: {from_csv: out/csv/hit_loads.csv, column: va, pick: final}
  F_pk: {from_csv: out/csv/hit_loads.csv, column: "Fz [kN]", pick: max}
bodies:
  b: {dof: translation, mass: 2 kg}
elements:
  push: {type: force, measure: b, csv: {file: out/csv/hit_loads.csv, y_col: Fz}}
outputs:
  vb: {body: b, quantity: velocity}
run: {t_end: 30 ms}
""")
    N = load_system(str(folder / "next.yaml")); o = N.run()
    h = S.run()
    assert N.nominal.ctx["v_last"] == pytest.approx(h["v"] / 1000, rel=1e-6)          # mm/s output vs m/s column
    assert N.nominal.ctx["F_pk"] == pytest.approx(h["F"] * 1000, rel=1e-6)             # kN column -> N
    assert o["vb"] == pytest.approx((2.0 - h["v"] / 1000) / 2.0, rel=1e-3)             # impulse from the table

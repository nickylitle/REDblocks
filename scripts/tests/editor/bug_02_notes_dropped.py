"""Notes the compiler accepts are dropped on round trip:
 (a) ground body `note` and `label` in 2D/3D (editor.html 1285, 1298: ground bodies only get {ground, feats});
 (b) force element (load node) `note`, `source`, `knockout` in 1D and 2D (1584/1609 only keep P/csv/pulse/label);
 (c) a param written as {value: "= expr", note} moves to `derived` and loses its note (1273);
 (d) output `note`, failure `reach_metric`, export `label`/`note` are not read by fromDoc (1639, 1646, 1707);
 (e) circuit `note`, part `note`/`source`, control-signal `note`, gravity planet `note` are dropped too (see probe_circuit_control.py)."""
import asyncio
from common import main, roundtrip, check

D2 = """simblocks: 1
name: t
dim: 2d
params:
  k: {value: 1000 N/m, note: param note}
  kk: {value: "= 2 * $k", note: derived note}
bodies:
  wall: {ground: true, label: the wall, note: ground note, points: {p: [0 mm, 0 mm]}}
  ball: {dof: planar, mass: 1 kg, inertia: 1 kg m^2, note: ball note}
elements:
  push: {type: force, measure: ball.y, P: 1 N, note: load note, source: hand, knockout: {expect: {Fmax: 0}}}
outputs:
  Fmax: {element: push, quantity: force, reduce: max, unit: N, note: output note}
failure:
  - {name: f1, metric: Fmax, limit: 2, fails_when: above, origin: "user: me", reach_metric: Fmax}
exports:
  - {name: csv, columns: [{body: ball, quantity: y}], label: export label, note: export note}
run:
  t_end: 0.01 s
"""

async def s(pg):
    doc, y = await roundtrip(pg, D2)
    ok = True
    ok &= check(doc["bodies"]["wall"].get("note") == "ground note", doc["bodies"]["wall"], "note kept", "(a) ground body note")
    ok &= check(doc["bodies"]["wall"].get("label") == "the wall", doc["bodies"]["wall"].get("label"), "the wall", "(a) ground body label")
    ok &= check(doc["bodies"]["ball"].get("note") == "ball note", doc["bodies"]["ball"].get("note"), "ball note", "(a) moving body note")
    p = doc["elements"]["push"]
    ok &= check(p.get("note") == "load note" and p.get("source") == "hand" and "knockout" in p, p, "note/source/knockout kept", "(b) force element")
    ok &= check(isinstance(doc.get("params", {}).get("kk"), dict) or isinstance(doc.get("derived", {}).get("kk"), dict),
                {"params.kk": doc.get("params", {}).get("kk"), "derived.kk": doc.get("derived", {}).get("kk")}, "note kept", "(c) = param with note")
    ok &= check(doc["outputs"]["Fmax"].get("note") == "output note", doc["outputs"]["Fmax"], "note kept", "(d) output note")
    ok &= check(doc["failure"][0].get("reach_metric") == "Fmax", doc["failure"][0], "reach_metric kept", "(d) failure reach_metric")
    ok &= check(doc["exports"][0].get("label") == "export label", doc["exports"][0], "label/note kept", "(d) export label")
    assert ok, "BUG: notes / annotations lost on round trip"
asyncio.run(main(s))

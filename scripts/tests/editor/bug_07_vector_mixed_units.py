"""commonUnit (editor.html 962-966) gives a bare number the unit of its neighbours: in a vector such as
at: [50 mm, 0.2] the compiler reads 0.2 as SI (0.2 m) but the editor stores it as '0.2 mm' on save (a factor 1000).
Affects 2D/3D body at/com/v0 vectors and point/circle/line/sphere/plane coordinates."""
import asyncio
from common import main, roundtrip, check
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from simblocks.compiler.units import parse_quantity

DOC = """simblocks: 1
name: t
dim: 2d
bodies:
  wall: {ground: true, points: {p: [50 mm, 0.2]}}
  ball: {dof: planar, mass: 1 kg, inertia: 1 kg m^2, at: [50 mm, 0.2], points: {c: [0, 0]}}
elements:
  pin: {type: pin, measure: [wall.p, ball.c], k: 1 kN/m}
outputs:
  y: {body: ball, quantity: y, reduce: final}
run:
  t_end: 0.01 s
"""

def si(v):
    return parse_quantity(v) if isinstance(v, str) else float(v)

async def s(pg):
    r = await pg.evaluate("async t => (await API.call('check', t))", DOC)
    print("compiler accepts:", r.get("ok"), r.get("error", ""))
    doc, y = await roundtrip(pg, DOC)
    at = doc["bodies"]["ball"]["at"]; p = doc["bodies"]["wall"]["points"]["p"]
    print("ball.at ->", at, "  wall.points.p ->", p)
    ok = check(abs(si(at[1]) - 0.2) < 1e-9, f"{at[1]} = {si(at[1])} m", "0.2 m", "body at[1]")
    ok &= check(abs(si(p[1]) - 0.2) < 1e-9, f"{p[1]} = {si(p[1])} m", "0.2 m", "point p[1]")
    assert ok, "BUG: bare SI number re-labelled with the neighbour's unit"
asyncio.run(main(s))

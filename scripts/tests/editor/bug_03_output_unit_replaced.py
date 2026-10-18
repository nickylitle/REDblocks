"""nodeHTML (editor.html 2513) silently REPLACES an output's unit with the default unit when the file's unit is not in the
editor's short UNITS list, even though the compiler accepts it (any product of known tokens: 'kN m', 'deg/s', 'lbf ft', 'km').
A failure-mode limit is read in the output's unit, so 'limit: 2' silently changes meaning from 2 kN m to 2 N m."""
import asyncio
from common import main, roundtrip, check

DOC = """simblocks: 1
name: t
dim: 1d
params:
  k: 10 N m/rad
bodies:
  rotor: {dof: rotation, inertia: 1 kg m^2, x0: 1 rad}
elements:
  shaft: {type: spring, measure: rotor, k: $k}
outputs:
  T:  {element: shaft, quantity: force, reduce: max, unit: kN m}
  w:  {body: rotor, quantity: velocity, reduce: max, unit: deg/s}
failure:
  - {name: shaft torque, metric: T, limit: 2, fails_when: above, origin: "user: me"}
run:
  t_end: 0.1 s
"""

async def s(pg):
    r = await pg.evaluate("async t => (await API.call('check', t))", DOC)
    print("compiler accepts the file as written:", r.get("ok"), r.get("error", ""))
    doc, y = await roundtrip(pg, DOC)
    ok = True
    ok &= check(doc["outputs"]["T"].get("unit") == "kN m", doc["outputs"]["T"].get("unit"), "kN m", "torque output unit")
    ok &= check(doc["outputs"]["w"].get("unit") == "deg/s", doc["outputs"]["w"].get("unit"), "deg/s", "speed output unit")
    print("failure limit still 2 ->", doc["failure"][0]["limit"], "(now in", doc["outputs"]["T"].get("unit"), "instead of kN m)")
    assert ok, "BUG: output unit replaced on load, failure limit changes meaning"
asyncio.run(main(s))

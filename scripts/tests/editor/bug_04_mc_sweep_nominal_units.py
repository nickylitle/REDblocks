"""Monte Carlo / Range nodes: when the nominal param's unit differs from the range's unit, fromDoc (1523, 1527) stores the
whole quantity ('0.005 m') in `nominal`, and toDoc (1346, 1348) joins it with the range unit again -> '0.005 m mm'
(a unit the compiler rejects). Also a sweep whose key is not in params is dropped entirely (1515 only walks params)."""
import asyncio
from common import main, roundtrip, check

DOC = """simblocks: 1
name: t
dim: 1d
params:
  k: 1000 N/m
  m: 1 kg
  x: 0.005 m
  g: 2 kN/m
bodies:
  mass: {dof: translation, mass: $m, x0: $x}
elements:
  spring: {type: spring, measure: mass, k: $k}
  spring2: {type: spring, measure: mass, k: $g}
outputs:
  F: {element: spring, quantity: force, reduce: max, unit: N}
uncertain:
  x: [4 mm, 6 mm]
study:
  monte_carlo: {n: 50, ranges: {x: [4 mm, 6 mm]}}
  sweep:
    g: [1 N/m, 2 N/m]
run:
  t_end: 0.1 s
"""

async def s(pg):
    r = await pg.evaluate("async t => (await API.call('check', t))", DOC)
    print("compiler accepts the file as written:", r.get("ok"), r.get("error", ""))
    doc, y = await roundtrip(pg, DOC)
    ok = True
    ok &= check(doc["params"]["x"] in ("0.005 m", 0.005, "5 mm"), doc["params"]["x"], "0.005 m", "mc nominal")
    ok &= check(doc["params"]["g"] in ("2 kN/m", 2000, "2000 N/m"), doc["params"]["g"], "2 kN/m", "sweep nominal")
    r2 = await pg.evaluate("async t => (await API.call('check', t))", y)
    ok &= check(r2.get("ok"), r2.get("error", "")[:200], "re-exported file compiles", "re-check")
    assert ok, "BUG: nominal gets two units"
asyncio.run(main(s))

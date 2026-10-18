"""Q1: a Goal seek node that is not wired yet is named in the status line (it is not in the file until wired).
F6: changing the default units from SI to US converts a 2D body's position vector and its points too."""
import asyncio
from common import main, load, export, check

DOC = """simblocks: 1
name: t
dim: 2d
units: SI
bodies:
  wall: {ground: true, points: {p: [0, 0]}}
  a: {dof: [x], mass: 1 kg, at: [0.05, 0], points: {tip: [0.01, 0]}}
elements:
  s: {type: spring, k: 1000, measure: a.x}
outputs:
  x: {body: a, quantity: x}
run: {t_end: 1 ms}
"""

async def s(pg):
    await load(pg, DOC)
    await pg.evaluate("() => { addNode('goal', {name:'goal'}, 600, 400); changed(); }")
    await pg.wait_for_timeout(1500)
    st = await pg.evaluate("() => document.getElementById('smsg').textContent")
    ok = check("not saved until wired" in st and "goal" in st, st, "status names the unwired goal node", "Q1")
    await pg.evaluate("() => { convertDefaults('SI', 'US'); S.meta.units = 'US'; changed(); }")
    await pg.wait_for_timeout(500)
    doc, y = await export(pg)
    at = doc["bodies"]["a"]["at"]; tip = doc["bodies"]["a"]["points"]["tip"]
    ok &= check(str(at[0]).endswith("in") and abs(float(str(at[0]).split()[0]) - 1.9685) < 1e-3, at, "[1.9685 in, 0 in]", "F6 at")
    ok &= check(str(tip[0]).endswith("in") and abs(float(str(tip[0]).split()[0]) - 0.393701) < 1e-4, tip, "[0.3937 in, 0 in]", "F6 point")
    assert ok
asyncio.run(main(s))

"""Switching the model's dim (toolbar) re-renders every output node; nodeHTML (editor.html 2504) replaces a body quantity
that the new dim does not list with ql[0], which is 'kinetic_energy' (the catalog list is sorted), and 2507/1329 then drop
the reduce and 2513 forces the unit to J. Switching back does not restore anything: an accidental 2D->1D->2D flip turns
'angle, min, deg' into 'kinetic_energy, J'. Even the intended 2D->1D switch should map angle/omega/x/vx to position/velocity."""
import asyncio
from common import main, export, load

DOC = """simblocks: 1
name: t
dim: 2d
bodies:
  wall: {ground: true, points: {p: [0, 0]}}
  bar: {dof: planar, mass: 1 kg, inertia: 0.01 kg m^2, angle0: 30 deg, points: {top: [0, 0]}}
elements:
  pin: {type: pin, measure: [wall.p, bar.top], k: 1e6 N/m}
outputs:
  ang_min: {body: bar, quantity: angle, reduce: min, unit: deg}
  spin_max: {body: bar, quantity: omega, reduce: max_abs, unit: rpm}
run:
  t_end: 0.1 s
"""

async def s(pg):
    await load(pg, DOC)
    await pg.select_option("#bDim", "1d"); await pg.wait_for_timeout(300)
    await pg.select_option("#bDim", "2d"); await pg.wait_for_timeout(300)
    doc, y = await export(pg)
    o = doc["outputs"]
    print("after 2d -> 1d -> 2d:", o)
    ok = o["ang_min"].get("quantity") == "angle" and o["ang_min"].get("reduce") == "min" and o["ang_min"].get("unit") == "deg"
    ok &= o["spin_max"].get("quantity") == "omega" and o["spin_max"].get("reduce") == "max_abs"
    print("PASS" if ok else "FAIL", "outputs survive a dim flip and back")
    print("   observed:", o)
    print("   expected: ang_min {quantity: angle, reduce: min, unit: deg}, spin_max {quantity: omega, reduce: max_abs, unit: rpm}")
    assert ok, "BUG: dim switch rewrites output quantity to kinetic_energy and drops reduce/unit"
asyncio.run(main(s))

"""The Gravity node is meant to be a single instance (pick() 3757 refuses a second one from the menu), but Ctrl+D / Ctrl+V
(insertFragment 3531) happily add gravity_2, gravity_3... toDoc (1266) only writes the first one found, so edits to the
copies are silently ignored."""
import asyncio
from common import main, load, export, OUT
DOC = """simblocks: 1
name: t
dim: 2d
gravity: 9.81 m/s^2
bodies:
  wall: {ground: true, points: {p: [0, 0]}}
  ball: {dof: planar, mass: 1 kg, inertia: 1 kg m^2, points: {c: [0, 0]}}
elements:
  pin: {type: pin, measure: [wall.p, ball.c], k: 1e6 N/m}
outputs:
  y: {body: ball, quantity: y, reduce: final}
run:
  t_end: 0.01 s
"""
async def s(pg):
    await load(pg, DOC)
    names = await pg.evaluate("() => { const g = S.nodes.find(n => n.kind === 'grav'); setSel(g.id); duplicateSel(); copySel(); pasteClip(); render(); return S.nodes.filter(n => n.kind === 'grav').map(n => n.data.name); }")
    await pg.evaluate("() => { const g2 = S.nodes.filter(n => n.kind === 'grav')[1]; if (g2) g2.data.g = '1.62 m/s^2'; }")
    doc, y = await export(pg)
    await pg.screenshot(path=OUT + "bug_12_gravity_dupes.png")
    print("gravity nodes:", names, "| file gravity:", doc.get("gravity"), "(gravity_2 was set to 1.62 m/s^2)")
    ok = len(names) == 1
    print("PASS" if ok else "FAIL", "one gravity node per model\n   observed:", names, "\n   expected: ['gravity'] (duplicate/paste should select the existing one like the menu does)")
    assert ok
asyncio.run(main(s))

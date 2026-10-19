"""ESC drag brake and gearbox efficiency survive the editor round trip, and the gearbox fields hide until ticked."""
import asyncio
from common import main, load, export, check

DOC = """simblocks: 1
name: t
dim: 1d
bodies:
  fw: {dof: rotation, inertia: 2e-4 kg m^2}
elements:
  b: {type: dashpot, c: 1e-6, measure: fw}
circuit:
  ground: gnd
  parts:
    pack: {type: voltage_source, between: [bus, gnd], V: 12 V}
    m1: {type: motor, between: [bus, gnd], kv: 1000 rpm/V, R: 0.2 ohm, gear: 10, eta: 0.82, measure: fw, esc: {throttle: -0.5, I_max: 15 A, regen: true, brake: true}}
    m2: {type: motor, between: [bus, gnd], kv: 1000 rpm/V, R: 0.2 ohm, measure: fw}
outputs:
  w: {body: fw, quantity: velocity, reduce: final}
run: {t_end: 10 ms}
"""

async def s(pg):
    await load(pg, DOC)
    doc, y = await export(pg)
    m1, m2 = doc["circuit"]["parts"]["m1"], doc["circuit"]["parts"]["m2"]
    ok = check(m1["esc"].get("brake") is True and str(m1["esc"].get("throttle")) == "-0.5" and float(m1.get("eta")) == 0.82,
               m1, "brake, reverse throttle and eta kept", "roundtrip")
    hid = await pg.evaluate("() => { const n = S.nodes.find(x => x.data.name === 'm2'); return !!document.querySelector(`.node[data-id='${n.id}'] input[data-k='f.eta']`); }")
    shown = await pg.evaluate("() => { const n = S.nodes.find(x => x.data.name === 'm1'); return !!document.querySelector(`.node[data-id='${n.id}'] input[data-k='f.eta'], .node[data-id='${n.id}'] [data-k='f.eta']`); }")
    ok &= check(not hid and shown, [hid, shown], "eta hidden on m2, shown on m1", "toggle")
    assert ok
asyncio.run(main(s))

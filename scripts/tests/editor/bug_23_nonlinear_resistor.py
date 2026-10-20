"""A nonlinear resistor (I-V table) can be added in the editor, its table typed in one box, and the file runs."""
import asyncio
from common import main, export, check
from simblocks.compiler import load_system

async def s(pg):
    await pg.evaluate("""() => { S.nodes = []; S.links = [];
      addNode('part', {name:'c', ptype:'capacitor', fields:{C:'10 nF', v0:'0.5 V'}}, 100, 100);
      addNode('part', {name:'nr', ptype:'nonlinear_resistor', fields:{iv:'[[-1, 1 mA], [1, -1 mA]]'}}, 500, 100); changed(); }""")
    await pg.wait_for_timeout(400)
    await pg.evaluate("""() => { const c = S.nodes.find(n => n.data.name==='c'), r = S.nodes.find(n => n.data.name==='nr');
      joinTerminals({node:c.id, key:'pa'}, {node:r.id, key:'pa'}, {clientX:0, clientY:0});
      joinTerminals({node:c.id, key:'pb'}, {node:r.id, key:'pb'}, {clientX:0, clientY:0}); }""")
    await pg.wait_for_timeout(400)
    doc, y = await export(pg)
    p = doc["circuit"]["parts"]["nr"]
    ok = check(p["type"] == "nonlinear_resistor" and "iv" in p, p, "part exported with its table", "export")
    txt = await pg.evaluate("() => document.querySelector('.node').closest('#world') ? [...document.querySelectorAll('.node')].map(n => n.textContent).join(' ') : ''")
    ok &= check("nonlinear resistor" in txt, txt[:200], "named on its node", "label")
    y2 = y + "\nrun: {t_end: 20 us}\n" if "run:" not in y else y
    doc["run"] = {"t_end": "20 us"}; doc.setdefault("outputs", {})["v"] = {"part": "c", "quantity": "voltage", "reduce": "final"}
    import yaml as _y
    o = load_system(_y.safe_dump(doc)).run()
    # negative resistor -1 kohm across 10 nF: v grows as exp(t / RC) = exp(20 us / 10 us)
    import math
    ok &= check(abs(o["v"] / (0.5 * math.exp(2.0)) - 1) < 2e-3, o["v"], f"0.5 e^2 = {0.5*math.exp(2):.4f} V", "runs")
    return ok

asyncio.run(main(s))

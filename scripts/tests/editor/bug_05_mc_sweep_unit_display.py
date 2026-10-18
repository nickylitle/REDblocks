"""Range / Monte Carlo node: the unit <select> (editor.html 2536, 2542) SHOWS the default unit of the fed field
(e.g. 'mm' in the mm system) while d.unit is still '' -> toDoc (1343-1348) writes the numbers with NO unit, which the
compiler reads as SI. The node says '4 .. 6 mm'; the file says 4 m .. 6 m (a factor 1000)."""
import asyncio
from common import main, export, check, OUT

async def s(pg):
    await pg.evaluate("""() => { S = newState(); S.meta.units = 'mm'; S.meta.dim = '1d';
        const b = addNode('body', {name:'mass', dof:'translation', mass:'1 kg', x0:''}, 400, 80);
        const m = addNode('mc', {name:'x', lo:'4', hi:'6', n:'20'}, 80, 80);
        const r = addNode('sweep', {name:'v', values:'1, 2, 3'}, 80, 300);
        connect({node:m.id, key:'out'}, {node:b.id, key:'x0'});
        connect({node:r.id, key:'out'}, {node:b.id, key:'v0'});
        render(); }""")
    await pg.wait_for_timeout(400)
    shown_mc = await pg.evaluate("() => { const n = S.nodes.find(x => x.kind==='mc'); return world.querySelector(`.node[data-id='${n.id}'] select[data-k='unit']`).value; }")
    shown_sw = await pg.evaluate("() => { const n = S.nodes.find(x => x.kind==='sweep'); return world.querySelector(`.node[data-id='${n.id}'] select[data-k='unit']`).value; }")
    doc, y = await export(pg)
    print("node shows unit:", shown_mc, shown_sw)
    print("file uncertain:", doc.get("uncertain"), " params:", doc.get("params"), " sweep:", doc.get("study", {}).get("sweep"))
    await pg.screenshot(path=OUT + "bug_05_mc_unit.png")
    ok = check(str(doc["uncertain"]["x"][0]).endswith(shown_mc), doc["uncertain"]["x"], f"[4 {shown_mc}, 6 {shown_mc}] as the node shows", "mc range unit")
    ok &= check(shown_sw in str(doc["study"]["sweep"]["v"]), doc["study"]["sweep"]["v"], f"values in {shown_sw} as the node shows", "sweep unit")
    assert ok, "BUG: node shows a unit the file does not carry"
asyncio.run(main(s))

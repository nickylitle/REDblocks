"""Dragging one part terminal onto another makes a net; current source and load are one Add entry with a direction switch."""
import asyncio
from common import main, export, check

async def s(pg):
    await pg.evaluate("""() => { S.nodes = []; S.links = [];
      addNode('part', {name:'cur', ptype:'current_source', fields:{I:'2 A'}}, 100, 100);
      addNode('part', {name:'r', ptype:'resistor', fields:{R:'10 ohm'}}, 500, 100); changed(); }""")
    await pg.wait_for_timeout(400)
    await pg.evaluate("""() => { const c = S.nodes.find(n => n.data.name==='cur'), r = S.nodes.find(n => n.data.name==='r');
      joinTerminals({node:c.id, key:'pa'}, {node:r.id, key:'pa'}, {clientX:0, clientY:0});
      joinTerminals({node:c.id, key:'pb'}, {node:r.id, key:'pb'}, {clientX:0, clientY:0}); }""")
    await pg.wait_for_timeout(400)
    doc, y = await export(pg)
    p = doc["circuit"]["parts"]
    ok = check(p["cur"]["between"] == p["r"]["between"] and len(set(p["r"]["between"])) == 2, p, "both parts on the same two nets", "join")
    ok &= check(doc["circuit"]["ground"] == "gnd" and p["r"]["between"][1] == "gnd", doc["circuit"], "minus side is gnd", "ground")
    sel = await pg.evaluate("() => !!document.querySelector('.node select[data-k=\"ptype\"]')")
    ok &= check(sel, sel, "direction switch on the current part", "switch")
    # the real mouse path: drag + of a new part onto + of the resistor
    await pg.evaluate("() => { addNode('part', {name:'c2', ptype:'capacitor', fields:{C:'1 uF'}}, 100, 500); changed(); }")
    await pg.wait_for_timeout(400)
    socks = await pg.query_selector_all('.sock.out[data-s="pa"]')
    boxes = [await x.bounding_box() for x in socks]
    src, dst = boxes[-1], boxes[1]
    await pg.mouse.move(src["x"] + 6, src["y"] + 6); await pg.mouse.down()
    await pg.mouse.move(dst["x"] + 6, dst["y"] + 6, steps=8); await pg.mouse.up()
    await pg.wait_for_timeout(400)
    doc, y = await export(pg)
    c2 = doc["circuit"]["parts"]["c2"]
    ok &= check(c2.get("between", [None])[0] == "n", c2, "c2 + joined net n by drag", "drag")
    assert ok
asyncio.run(main(s))

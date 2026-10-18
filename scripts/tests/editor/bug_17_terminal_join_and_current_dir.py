"""Dragging one part terminal onto another makes a net; current source and load are one Add entry with a direction switch."""
import asyncio
from common import main, export, check, load

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
    nb = await pg.evaluate("() => document.querySelectorAll('.node .hd').length - document.querySelectorAll('.node').length + [...document.querySelectorAll('.node')].filter(e => (byId(e.dataset.id) || {}).kind === 'net').length")
    ok &= check(nb == 0, nb, "no net blocks on the canvas", "hidden")
    w = await pg.evaluate("() => document.querySelectorAll('path[data-net]').length")
    ok &= check(w == 3, w, "3 terminal-to-terminal wires", "wires")
    await pg.screenshot(path="/tmp/claude-0/-home-claude/9883ddf4-3cb2-512a-abcd-30edf08680bb/scratchpad/hidden_nets.png")
    # the file keeps them hidden after a reload
    await load(pg, y); await pg.wait_for_timeout(400)
    nn = await pg.evaluate("() => S.nodes.filter(n => n.kind === 'net' && !n.data.hidden).length")
    ok &= check(nn == 0, nn, "still hidden after reload", "reload")
    # a file with no layout: nets come in hidden; double-click a wire shows the block
    await load(pg, "simblocks: 1\nname: t\ndim: 1d\ncircuit:\n  ground: gnd\n  parts:\n    s: {type: current_source, between: [a, gnd], I: 2 A}\n    r: {type: resistor, between: [a, gnd], R: 10 ohm}\noutputs:\n  v: {part: r, quantity: voltage, reduce: max}\nrun: {t_end: 1 ms}\n")
    await pg.wait_for_timeout(400)
    nn = await pg.evaluate("() => S.nodes.filter(n => n.kind === 'net' && !n.data.hidden).length")
    ok &= check(nn == 0, nn, "no-layout file: nets hidden", "nolayout")
    await pg.evaluate("() => { const net = S.nodes.find(n => n.kind === 'net' && n.data.name === 'a'); showNet(net); }")
    await pg.wait_for_timeout(300)
    doc, y = await export(pg)
    ok &= check(doc["circuit"]["parts"]["r"]["between"] == ["a", "gnd"] and doc["editor"].get("hidden_nets") == ["gnd"], doc["editor"].get("hidden_nets"), "a shown, gnd hidden, wiring same", "show")
    assert ok
asyncio.run(main(s))

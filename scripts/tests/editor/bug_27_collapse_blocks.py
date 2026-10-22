"""Blocks collapse to their title bar (button, C for the selection, Shift+C for all, Organize menu); wires meet the
title bar; the collapsed set is saved in the file and comes back when it opens."""
import asyncio, os
from common import main, check, load, export
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "learn", "1d_4_flywheel_circuit.yaml")

async def s(pg):
    ok = True
    await load(pg, open(EX).read())
    nid = await pg.evaluate("() => S.nodes.find(n => n.kind === 'part' || n.kind === 'element').id")
    h0 = await pg.evaluate(f"() => document.querySelector('.node[data-id=\"{nid}\"]').offsetHeight")
    await pg.evaluate(f"() => document.querySelector('.node[data-id=\"{nid}\"] [data-act=fold]').click()")
    await pg.wait_for_timeout(300)
    st = await pg.evaluate(f"() => {{ const el = document.querySelector('.node[data-id=\"{nid}\"]'); const l = S.links.find(l => l.to.node === '{nid}' || l.from.node === '{nid}');"
                           f" const p = l ? (l.to.node === '{nid}' ? sockPos(l.to.node, l.to.key, 'in') : sockPos(l.from.node, l.from.key, 'out')) : null;"
                           f" const hd = el.querySelector('.hd').getBoundingClientRect(); return [el.classList.contains('fold'), el.offsetHeight, !!p]; }}")
    ok &= check(st[0] and st[1] < h0 / 2 and st[2], (h0, st), "collapsed to its title bar, wires still placed", "fold-button")
    doc, y = await export(pg)
    ok &= check(len((doc.get("editor") or {}).get("collapsed", [])) == 1, (doc.get("editor") or {}).get("collapsed"), "saved in the file", "saved")
    await load(pg, y)
    back = await pg.evaluate("() => S.nodes.filter(n => n.fold).length")
    ok &= check(back == 1, back, "comes back collapsed when opened", "reload")
    await pg.evaluate("() => setSel([])"); await pg.keyboard.press("Shift+C"); await pg.wait_for_timeout(300)
    allf = await pg.evaluate("() => [S.nodes.filter(n => n.kind !== 'note' && n.kind !== 'image').every(n => n.fold), document.querySelectorAll('.node.fold').length]")
    ok &= check(allf[0] and allf[1] > 3, allf, "Shift+C collapses every block", "fold-all")
    q = os.path.join(os.path.dirname(EX), "3d_2_quadcopter.yaml")      # name references ($x in a field) too
    await load(pg, open(q).read(), os.path.dirname(q)); await pg.evaluate("() => setSel([])")
    await pg.keyboard.press("Shift+C"); await pg.wait_for_timeout(400)
    far = await pg.evaluate("() => { const ys = [...document.querySelectorAll('#wires path, #wiresTop path, #wires circle')].map(p => { const b = p.getBBox(); return [b.y, b.y + b.height]; });"
                            " const nodes = [...document.querySelectorAll('.node')].map(e => [parseFloat(e.style.top), parseFloat(e.style.top) + e.offsetHeight]);"
                            " const lo = Math.min(...nodes.map(n => n[0])) - 200, hi = Math.max(...nodes.map(n => n[1])) + 200;"
                            " return [ys.length, ys.filter(([a, b]) => a < lo || b > hi).length]; }")
    ok &= check(far[0] > 10 and far[1] == 0, far, "every wire stays among the blocks when all are collapsed", "wires-in-place")
    await pg.keyboard.press("Shift+C"); await pg.wait_for_timeout(300)
    none = await pg.evaluate("() => document.querySelectorAll('.node.fold').length")
    ok &= check(none == 0, none, "Shift+C again expands them", "unfold-all")
    return ok

asyncio.run(main(s))

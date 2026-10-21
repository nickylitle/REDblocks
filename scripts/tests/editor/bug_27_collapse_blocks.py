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
    await pg.keyboard.press("Shift+C"); await pg.wait_for_timeout(300)
    none = await pg.evaluate("() => document.querySelectorAll('.node.fold').length")
    ok &= check(none == 0, none, "Shift+C again expands them", "unfold-all")
    return ok

asyncio.run(main(s))

"""A Note (or Image) folded into a group is written into editor.groups[].members as 'undefined:undefined' (nodeKey 1452 has
no namespace for note/image) and on reload byKey matches whichever note happens to compare first. Cosmetic junk in the
saved file; wrong membership once a model has two notes."""
import asyncio
from common import main, export
async def s(pg):
    await pg.evaluate("""() => { S = newState(); S.meta.dim = '1d';
        const k = addNode('value', {name:'k', value:'2 N/m'}, 40, 40);
        const n1 = addNode('note', {text:'first'}, 40, 300), n2 = addNode('note', {text:'second'}, 40, 500);
        render(); setSel([k.id, n2.id]); makeGroup(); }""")
    doc, y = await export(pg)
    members = doc["editor"]["groups"][0]["members"]
    print("group members in file:", members)
    await pg.evaluate("async t => { await loadText(t, null, null); }", y); await pg.wait_for_timeout(300)
    back = await pg.evaluate("() => S.groups[0].members.map(id => { const n = byId(id); return n.kind + ':' + (n.data.name || n.data.text); })")
    print("group members after reload:", back)
    ok = "undefined:undefined" not in members and "note:second" in back and "note:first" not in back
    print("PASS" if ok else "FAIL", "note membership round-trips\n   observed:", members, back, "\n   expected: a real key for the note, and the SAME note (second) back in the group")
    assert ok
asyncio.run(main(s))

"""Auto-named hidden nets do not show on terminal labels; the Texas theme is gone; the app is called REDblocks."""
import asyncio
from common import main, check

async def s(pg):
    await pg.evaluate("""() => { S.nodes = []; S.links = [];
      addNode('part', {name:'cur', ptype:'current_source', fields:{I:'2 A'}}, 100, 100);
      addNode('part', {name:'r', ptype:'resistor', fields:{R:'10 ohm'}}, 500, 100); changed(); }""")
    await pg.wait_for_timeout(400)
    await pg.evaluate("""() => { const c = S.nodes.find(n => n.data.name==='cur'), r = S.nodes.find(n => n.data.name==='r');
      joinTerminals({node:c.id, key:'pa'}, {node:r.id, key:'pa'}, {clientX:0, clientY:0});
      joinTerminals({node:c.id, key:'pb'}, {node:r.id, key:'pb'}, {clientX:0, clientY:0}); }""")
    await pg.wait_for_timeout(400)
    labs = await pg.evaluate("() => [...document.querySelectorAll('.node .row')].map(e => e.textContent.trim()).filter(t => /terminal/.test(t))")
    ok = check(any(t.startswith("+ terminal") for t in labs) and not any(" · n" in t for t in labs), labs, "no auto net names on terminals", "labels")
    ok &= check(any("· gnd" in t for t in labs), labs, "ground still marked", "gnd")
    tx = await pg.evaluate("() => [!!THEMES.texas, document.title, curLook().id]")
    ok &= check(not tx[0] and tx[1].startswith("REDblocks"), tx, "no Texas theme, title REDblocks", "texas+name")
    await pg.evaluate("() => { SETTINGS.theme = 'texas'; applyColors(); }")
    fb = await pg.evaluate("() => themeId()")
    ok &= check(fb == "redline", fb, "a saved Texas choice falls back to Redline", "fallback")
    await pg.evaluate("() => settingsShow()"); await pg.wait_for_timeout(600)
    await pg.screenshot(path="/tmp/claude-0/-home-claude/9883ddf4-3cb2-512a-abcd-30edf08680bb/scratchpad/settings_themes.png")
    return ok

asyncio.run(main(s))

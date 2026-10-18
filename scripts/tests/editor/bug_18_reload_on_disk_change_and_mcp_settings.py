"""The open file changes on disk (an assistant edited it through MCP): the editor reloads it, or warns when there are
unsaved edits. Settings shows the AI assistant (MCP) setup."""
import asyncio, os, tempfile, time
from common import main, check

TXT = """simblocks: 1
name: t
dim: 1d
params: {k: 1000}
bodies:
  m: {dof: translation, mass: 1 kg, x0: 1 mm}
elements:
  s: {type: spring, k: $k, measure: m}
outputs:
  x: {body: m, quantity: position}
run: {t_end: 1 ms}
"""

async def s(pg):
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.yaml"); open(p, "w").write(TXT)
    await pg.evaluate("async ([t, p]) => { await loadText(t, p); }", [TXT, p])
    await pg.wait_for_timeout(600)
    time.sleep(1.1); open(p, "w").write(TXT.replace("k: 1000", "k: 2500"))
    await pg.wait_for_timeout(3000)
    k = await pg.evaluate("() => (S.nodes.find(n => n.kind === 'value' && n.data.name === 'k') || {data:{}}).data.value")
    ok = check(str(k).strip() == "2500", k, "reloaded value 2500", "reload")
    st = await pg.evaluate("() => document.getElementById('smsg').textContent")
    ok &= check("changed on disk" in st, st, "status says it reloaded", "status")
    # unsaved edits: warn, do not reload
    await pg.evaluate("() => { const n = S.nodes.find(n => n.kind === 'value' && n.data.name === 'k'); n.data.value = '777'; changed(); }")
    time.sleep(1.1); open(p, "w").write(TXT.replace("k: 1000", "k: 9999"))
    await pg.wait_for_timeout(3000)
    k = await pg.evaluate("() => S.nodes.find(n => n.kind === 'value' && n.data.name === 'k').data.value")
    st = await pg.evaluate("() => document.getElementById('smsg').textContent")
    ok &= check(str(k) == "777" and "unsaved edits are still here" in st, [k, st], "kept 777 and warned", "dirty")
    await pg.evaluate("() => settingsShow()"); await pg.wait_for_timeout(800)
    box = await pg.evaluate("() => document.getElementById('mcpBox').textContent")
    ok &= check("claude mcp add redblocks" in box and "mcpServers" in box, box[:200], "setup commands shown", "settings")
    assert ok
asyncio.run(main(s))

"""An assistant asks the running editor to open a model (MCP open_in_editor): the editor opens it within a few seconds."""
import asyncio, os, tempfile, threading
from common import main, check
from simblocks.mcp_server import tool_open_in_editor

TXT = open(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "learn", "1d_1_mass_spring_damper.yaml")).read()

async def s(pg):
    d = tempfile.mkdtemp(); p = os.path.join(d, "asked.yaml"); open(p, "w").write(TXT)
    res = {}
    th = threading.Thread(target=lambda: res.update(tool_open_in_editor(p))); th.start()
    await pg.wait_for_timeout(4500); th.join()
    fp = await pg.evaluate("() => filePath")
    ok = check(fp == p and res.get("opened") == p, [fp, res], "editor opened the asked file", "open")
    assert ok
asyncio.run(main(s))

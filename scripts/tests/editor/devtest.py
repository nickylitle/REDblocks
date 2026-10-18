"""Drive the editor in headless Chromium with the real Api bridged in (stands in for the pywebview window).
Used by the editor regression scripts in this folder (each bug_*.py exits 1 if its bug is back)."""
import asyncio, json, os, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from simblocks.app.api import Api
from playwright.async_api import async_playwright
OUT = os.path.join(tempfile.gettempdir(), "simblocks_editor_tests") + os.sep
os.makedirs(OUT, exist_ok=True)
HTML = "file://" + os.path.join(ROOT, "simblocks", "app", "editor.html").replace("\\", "/")
api = Api()


async def main(script):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1600, "height": 950})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
        async def bridge(m, args):
            r = await asyncio.to_thread(getattr(api, m), *json.loads(args))
            return json.dumps(r, default=str)
        await pg.expose_function("__devApi", bridge)
        await pg.goto(HTML); await pg.wait_for_timeout(2500)
        await pg.evaluate("() => { const m = document.getElementById('modal'); if (m && m.style.display === 'flex') document.getElementById('mdNo').click(); }")
        await script(pg)
        for l in logs:
            print(l)
        await b.close()

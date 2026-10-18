"""Settings > colours: the 'Electrical wires and parts' picker (token s-elec) gets value "undefined" for every built-in theme.
settingsShow (editor.html 5414-5416) reads t = blueprint.t + theme.t, but s-elec is only derived at apply time
(5354: T["s-elec"] || T.pass). The browser logs a format warning and the swatch shows black instead of the live colour."""
import asyncio, json
from common import main, OUT

async def s(pg):
    warns = []
    pg.on("console", lambda m: warns.append(m.text) if "rrggbb" in m.text else None)
    await pg.click("#bSettings"); await pg.wait_for_timeout(500)
    r = await pg.evaluate("""() => { const i = document.querySelector('#setColors input[data-tok="s-elec"]');
        return {attr: i.getAttribute('value'), live: getComputedStyle(document.documentElement).getPropertyValue('--s-elec').trim(), shown: i.value}; }""")
    print("picker value attr:", r["attr"], " picker shows:", r["shown"], " live --s-elec:", r["live"], " console warnings:", len(warns))
    el = await pg.query_selector('#setColors input[data-tok="s-elec"]')
    await (await el.evaluate_handle("e => e.closest('.cgrid')")).as_element().screenshot(path=OUT + "bug_10_settings_elec.png")
    ok = r["attr"] != "undefined" and r["shown"].lower() == r["live"].lower()
    print("PASS" if ok else "FAIL", "s-elec picker shows the live colour\n   observed:", r, "\n   expected: value attribute = live --s-elec")
    assert ok
asyncio.run(main(s))

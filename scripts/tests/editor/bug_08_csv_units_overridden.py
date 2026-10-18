"""csvOut (editor.html 2112) always writes x_unit / y_unit, defaulting to 's' / 'N' when the file gave none. The compiler
(loadtable.py 200-201) takes a missing x_unit / y_unit from the CSV header ('t [ms]', 'F [kN]'), so a file that was correct
becomes wrong by 1000x after save. Also csvIn (2107) ignores the inline `x:` / `y:` lists form of csv, which is lost."""
import asyncio, os
from common import main, roundtrip, check, OUT

import tempfile; HERE = tempfile.mkdtemp(prefix="sb_csv_")          # the CSV lives outside the repo
open(os.path.join(HERE, "pulse.csv"), "w").write("t [ms],F [kN]\n0,0\n2.5,2\n5,0\n100,0\n")

DOC = """simblocks: 1
name: t
dim: 1d
bodies:
  mass: {dof: translation, mass: 1 kg}
elements:
  push: {type: force, measure: mass, csv: {file: pulse.csv}}
  inline: {type: force, measure: mass, csv: {x: [0, 1, 2], y: [0, 5, 0], x_unit: ms, y_unit: N}}
outputs:
  F: {element: push, quantity: force, reduce: max_abs, unit: N}
  Fi: {element: inline, quantity: force, reduce: max, unit: N}
run:
  t_end: 0.02 s
"""

async def s(pg):
    await pg.evaluate("async d => { await API.call('set_model_dir', d + '/_x.yaml'); }", HERE)
    DOC1 = DOC.replace("  inline: {type: force, measure: mass, csv: {x: [0, 1, 2], y: [0, 5, 0], x_unit: ms, y_unit: N}}\n", "").replace("  Fi: {element: inline, quantity: force, reduce: max, unit: N}\n", "")
    r1 = await pg.evaluate("async t => (await API.call('run', t))", DOC1)
    print("original run:", r1.get("ok"), r1.get("error", ""), (r1.get("outputs") or r1.get("results") or {}))
    doc1, y1 = await roundtrip(pg, DOC1, HERE)
    r2 = await pg.evaluate("async t => (await API.call('run', t))", y1)
    doc, y = await roundtrip(pg, DOC, HERE)
    print("exported csv:", doc["elements"]["push"]["csv"], "|", doc["elements"]["inline"].get("csv"))
    print("re-exported run:", r2.get("ok"), r2.get("error", ""), (r2.get("outputs") or r2.get("results") or {}))
    o1 = (r1.get("outputs") or r1.get("results") or {}); o2 = (r2.get("outputs") or r2.get("results") or {})
    ok = check(doc["elements"]["push"]["csv"].get("x_unit") in (None, "ms") and doc["elements"]["push"]["csv"].get("y_unit") in (None, "kN"),
               doc["elements"]["push"]["csv"], "no units written (header has them) or ms/kN", "csv units")
    ok &= check("x" in (doc["elements"]["inline"].get("csv") or {}), doc["elements"]["inline"].get("csv"), "inline x/y lists kept", "inline csv")
    if r1.get("ok") and r2.get("ok"):
        ok &= check(abs(float(o1.get("F", 0)) - float(o2.get("F", 1))) < 1e-6, f"F before {o1.get('F')} after {o2.get('F')}", "same peak force", "peak force")
    assert ok, "BUG: csv units overridden / inline csv lost"
asyncio.run(main(s))

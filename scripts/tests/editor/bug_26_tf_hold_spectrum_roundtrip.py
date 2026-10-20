"""Transfer-function and hold signals and spectrum exports survive the editor (load, export, same file meaning)."""
import asyncio
from common import main, check, load, export
from simblocks.compiler import load_system

Y = """simblocks: 1
name: t
dim: 1d
bodies: {m: {dof: translation, mass: 1 kg, x0: 1 mm}}
elements:
  s: {type: spring, k: 400 N/m, measure: m}
  push: {type: force, measure: m, P: {control: y, init: 0}}
control:
  u: "= 1 if t > 0.1 else 0"
  y: {transfer: u, num: [1, 10], den: [0.02, 1, 100]}
  held: {hold: y, period: 5 ms}
outputs:
  x: {body: m, quantity: position, reduce: max}
  h: {control: held, reduce: max}
exports:
  - {name: srs, spectrum: {kind: srs, f_min: 1 Hz, f_max: 200 Hz, Q: 10}, columns: [{body: m, quantity: acceleration}], folder: out}
run: {t_end: 1 s}
"""

async def s(pg):
    await load(pg, Y)
    doc, y2 = await export(pg)
    c = doc["control"]
    ok = check(c["y"].get("transfer") == "u" and c["y"]["num"] == [1, 10] and c["y"]["den"] == [0.02, 1, 100], c["y"], "transfer kept", "tf")
    ok &= check(c["held"].get("hold") == "y" and str(c["held"]["period"]).replace(" ", "") == "5ms", c["held"], "hold kept", "hold")
    sp = doc["exports"][0].get("spectrum")
    ok &= check(isinstance(sp, dict) and sp["kind"] == "srs" and sp.get("Q") == 10, sp, "spectrum export kept", "spectrum")
    o1 = load_system(Y).run(); o2 = load_system(y2).run()
    ok &= check(abs(o1["x"] - o2["x"]) < 1e-12 and abs(o1["h"] - o2["h"]) < 1e-12, (o1, o2), "same results", "run")
    return ok

asyncio.run(main(s))

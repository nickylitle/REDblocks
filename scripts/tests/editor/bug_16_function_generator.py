"""Function generator: a wave on a current load and a load profile on a battery survive the editor round trip,
and the part draws its preview."""
import asyncio
from common import main, load, export, check

DOC = """simblocks: 1
name: t
dim: 1d
circuit:
  ground: gnd
  parts:
    pack: {type: battery, between: [vb, gnd], cells: 6, capacity: 2200 mAh, I_load: {shape: pulse, amplitude: 20 A, offset: 1 A, frequency: 5 Hz, duty: 0.2}}
    ld: {type: current_load, between: [vb, gnd], wave: {shape: sine, amplitude: 2 A, offset: 3 A, frequency: 50 Hz, phase: 90 deg}}
    r: {type: resistor, between: [vb, gnd], R: 10 ohm}
outputs:
  i: {part: ld, quantity: current, reduce: max}
run: {t_end: 50 ms}
"""

async def s(pg):
    await load(pg, DOC)
    n = await pg.evaluate("() => document.querySelectorAll('.csvmini[data-wv]').length")
    ok = check(n == 2, n, "2 waveform previews", "preview")
    doc, y = await export(pg)
    p = doc["circuit"]["parts"]
    ok &= check(p["ld"].get("wave", {}).get("shape") == "sine" and str(p["ld"]["wave"].get("phase")) == "90 deg"
                and str(p["ld"]["wave"].get("frequency")) == "50 Hz", p["ld"], "sine wave kept", "wave")
    il = p["pack"].get("I_load")
    ok &= check(isinstance(il, dict) and il.get("shape") == "pulse" and str(il.get("duty")) == "0.2", il, "pulse load kept", "I_load")
    ok &= check("I" not in p["ld"], p["ld"], "no stray I", "fields")
    assert ok
asyncio.run(main(s))

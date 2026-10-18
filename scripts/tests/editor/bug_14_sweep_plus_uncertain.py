"""A parameter that is both swept (study.sweep) and uncertain (uncertain / monte_carlo.ranges) loses its uncertainty:
fromDoc (1517-1529) makes it a Range node and never looks at mcr[k]; unc_extra (1511) only keeps keys that are not params."""
import asyncio
from common import main, roundtrip, MIN1D
DOC = MIN1D + """uncertain:
  k: [900 N/m, 1100 N/m]
study:
  sweep: {k: [500 N/m, 1000 N/m, 2000 N/m]}
  monte_carlo: {n: 20, ranges: {k: [900 N/m, 1100 N/m]}}
"""
async def s(pg):
    r = await pg.evaluate("async t => (await API.call('check', t))", DOC)
    print("compiler accepts:", r.get("ok"), r.get("error", ""))
    doc, y = await roundtrip(pg, DOC)
    print("uncertain:", doc.get("uncertain"), " study:", doc.get("study"))
    ok = "k" in (doc.get("uncertain") or {}) and "monte_carlo" in (doc.get("study") or {})
    print("PASS" if ok else "FAIL", "uncertain + sweep on the same param survive\n   observed:", doc.get("uncertain"), doc.get("study"), "\n   expected: uncertain.k and study.monte_carlo kept alongside the sweep")
    assert ok
asyncio.run(main(s))

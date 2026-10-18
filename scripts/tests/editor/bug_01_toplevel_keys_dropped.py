"""Top-level keys the compiler accepts (limits, knockouts, report, assumptions, sources, notes) are dropped by fromDoc/toDoc.
editor.html 1505 builds a `known` list but never stores the unknown keys in S.meta.extra (which toDoc 1436 would re-emit)."""
import asyncio
from common import main, roundtrip, check, MIN1D

DOC = MIN1D + """limits:
  - {name: no stiffness, set: {k: 0}, expect: {F: 0}}
knockouts:
  weak: {set: {k: 1}, expect: {F: 0}}
report:
  plots: [{title: Spring force, series: [{ref: spring.force, unit: N}]}]
assumptions: ["linear spring", "no damping"]
sources: ["Roark 7e"]
notes: ["a note for the reader"]
"""

async def s(pg):
    doc, y = await roundtrip(pg, DOC)
    ok = True
    for k in ("limits", "knockouts", "report", "assumptions", "sources", "notes"):
        ok &= check(k in doc, f"'{k}' in exported doc: {k in doc}", f"'{k}' preserved", k)
    assert ok, "BUG: top-level keys lost on round trip"
asyncio.run(main(s))

"""Shared helpers for the audit repro scripts."""
import sys, yaml
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devtest import main, OUT   # noqa

async def load(pg, text, d=None):
    await pg.evaluate("async ([t, d]) => { await loadText(t, null, d); }", [text, d])
    await pg.wait_for_timeout(500)

async def export(pg):
    y = await pg.evaluate("async () => (await API.call('export_yaml', toDoc())).text")
    return yaml.safe_load(y), y

async def roundtrip(pg, text, d=None):
    await load(pg, text, d)
    return await export(pg)

def check(cond, observed, expected, label=""):
    tag = "PASS" if cond else "FAIL"
    print(f"{tag} {label}\n   observed: {observed}\n   expected: {expected}")
    return cond

MIN1D = """simblocks: 1
name: t
dim: 1d
params:
  k: 1000 N/m
  m: 1 kg
bodies:
  mass: {dof: translation, mass: $m, x0: 1 mm}
elements:
  spring: {type: spring, measure: mass, k: $k}
outputs:
  F: {element: spring, quantity: force, reduce: max, unit: N}
run:
  t_end: 0.1 s
"""

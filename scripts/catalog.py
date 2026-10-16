"""Catalog CLI.
    python scripts/catalog.py                      # summary by domain/category
    python scripts/catalog.py --find plates        # text search over names, summaries, tags
    python scripts/catalog.py --domain structures --category beams
    python scripts/catalog.py --show cantilever_end_load   # full docstring + signature + source
    python scripts/catalog.py --materials          # materials keys
    python scripts/catalog.py --write              # regenerate references/CATALOG.md and MATERIAL_CATALOG.md
"""
import argparse, collections, inspect, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simblocks as sb
from simblocks import materials as M

ap = argparse.ArgumentParser()
ap.add_argument("--find"); ap.add_argument("--domain"); ap.add_argument("--category")
ap.add_argument("--kind"); ap.add_argument("--show"); ap.add_argument("--materials", action="store_true")
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

if a.write:
    ref = os.path.join(os.path.dirname(HERE), "references")
    open(os.path.join(ref, "CATALOG.md"), "w").write(sb.catalog_markdown())
    open(os.path.join(ref, "MATERIAL_CATALOG.md"), "w").write(M.catalog_markdown())
    print(f"wrote CATALOG.md ({len(sb.REGISTRY)} blocks) and MATERIAL_CATALOG.md ({len(M.catalog())} materials)")
elif a.show:
    b = sb.REGISTRY[a.show]; f = getattr(b.func, "py_func", b.func)
    print(f"{b.name}{inspect.signature(f)}\n  path: {b.path}\n  kind: {b.kind}\n  source: {b.source}\n  module: {f.__module__}\n")
    print(inspect.getdoc(f))
elif a.materials:
    for k, m in M.catalog().items():
        print(f"{k:20s} {m.name}")
elif a.find or a.domain or a.category or a.kind:
    for b in sb.find(domain=a.domain, category=a.category, kind=a.kind, text=a.find):
        print(f"{b.name:38s} {b.path:60s} [{b.kind}] {b.summary[:90]}")
else:
    c = collections.Counter((b.domain, b.category) for b in sb.REGISTRY.values())
    cur = None
    for (d, cat), n in sorted(c.items()):
        if d != cur:
            print(f"\n{d}"); cur = d
        print(f"  {cat:22s} {n}")
    print(f"\n{len(sb.REGISTRY)} blocks, {len(M.catalog())} materials")

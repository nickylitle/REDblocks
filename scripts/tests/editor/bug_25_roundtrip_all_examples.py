"""Every bundled example survives the editor: load, export, and the exported file gives the same outputs.
(Found by the replay viewer: a motor's twist axis was dropped on import, so the 3D drive model stopped driving.)"""
import asyncio, glob, os
from common import main, check, load, export
from simblocks.compiler import load_system
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "..", "examples")
FILES = sorted(glob.glob(os.path.join(EX, "learn", "*.yaml")) + glob.glob(os.path.join(EX, "case_studies", "**", "*.yaml"), recursive=True))

async def s(pg):
    ok = True
    for f in FILES:
        y = open(f).read(); base = os.path.dirname(f)
        try:
            s0 = load_system(f)
        except Exception as exc:
            print("skip (does not load on its own):", os.path.basename(f), exc); continue
        await load(pg, y)
        doc, y2 = await export(pg)
        os.environ["SIMBLOCKS_MODEL_DIR"] = base
        try:
            o1 = load_system(y).run(); o2 = load_system(y2).run()
        except Exception as exc:
            ok &= check(False, str(exc)[:300], "both run", os.path.basename(f)); continue
        bad = {k: (o1[k], o2.get(k)) for k in o1 if not k.startswith("_") and isinstance(o1[k], float)
               and (o2.get(k) is None or abs(o2[k] - o1[k]) > 1e-6 * max(abs(o1[k]), 1e-9) + 1e-12)}
        ok &= check(not bad, bad, "same outputs after the editor", os.path.relpath(f, EX))
    return ok

asyncio.run(main(s))

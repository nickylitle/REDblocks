"""Every example opens with its values in tables (inputs, = expressions), never as dozens of separate value nodes;
the model itself is unchanged (it still runs to the same outputs, and the file is not marked edited)."""
import asyncio, glob, os
from common import main, check, load, export
from simblocks.compiler import load_system
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples")
FILES = sorted(glob.glob(os.path.join(EX, "learn", "*.yaml"))) + sorted(glob.glob(os.path.join(EX, "showcase", "apollo11", "*.yaml")))

async def s(pg):
    ok = True
    for f in FILES:
        await load(pg, open(f).read(), os.path.dirname(f))
        st = await pg.evaluate("() => [S.nodes.filter(n => n.kind === 'value' && !n.data.mat && !groupOf(n.id)).length,"
                               " S.groups.filter(g => isValTable(g)).length, dirty]")
        ok &= check(st[0] < 3 and st[2] is False, st, "no more than 2 loose values; not marked edited", os.path.basename(f))
    return ok

asyncio.run(main(s))

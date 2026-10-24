"""The bullwhip showcase in Animate: seen square onto its plane and fitted, drawn at true size (no deformation
magnified), one trail (the tip's, glowing with speed), the sonic boom's ring spreading at 343 m/s, slow-mo on by
itself, and the whole run still taking the seconds asked."""
import asyncio, os
from common import main, check, load
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "showcase", "bullwhip", "bullwhip_crack.yaml")

async def s(pg):
    ok = True
    await pg.set_viewport_size({"width": 1400, "height": 820})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    await load(pg, open(EX).read(), os.path.dirname(EX))
    await pg.evaluate("() => { document.getElementById('bLive').checked = false; }")
    await pg.evaluate("() => animOpen()")
    for _ in range(900):
        await pg.wait_for_timeout(200)
        if await pg.evaluate("() => AV.open && !!AV.S"):
            break
    v = await pg.evaluate("() => [AV.orbit.az, AV.orbit.el, AV.follow, AV.magF || 1, AV.trails.length, !!AV.trails[0].line.geometry.attributes.color]")
    ok &= check(abs(v[0] + 1.5708) < 1e-3 and v[1] < 0.3 and v[2] == -1, v[:3], "square onto the whip's plane, nothing followed", "view")
    ok &= check(v[3] == 1 and v[4] == 1 and v[5], v[3:], "true size; one trail, coloured by speed", "trail")
    sm = await pg.evaluate("() => { const b = document.getElementById('avFitS'); b.value = '12'; b.oninput();"
                           " return [document.getElementById('avSlowmo').checked, AV.rate, AV.swT, Math.min(...AV.sw)]; }")
    ok &= check(sm[0] and abs(sm[1] - sm[2] / 12) < 1e-12 and sm[3] < 0.2, sm, "slow-mo on by itself, the run still takes 12 s", "slowmo")
    bm = await pg.evaluate("() => { const o = AV.booms[0]; AV.tcur = o.bm.t + 0.002; avPose(); const a = [o.g.visible, o.g.scale.x];"
                           " AV.tcur = o.bm.t - 0.001; avPose(); return [AV.booms.length, ...a, o.g.visible]; }")
    ok &= check(bm[0] == 1 and bm[1] and abs(bm[2] - 0.686) < 1e-3 and not bm[3], bm, "the boom ring is 343 m/s x 2 ms across after, gone before", "boom")
    ok &= check(not errs, errs, "no page errors", "errors")
    return ok

asyncio.run(main(s))

"""Animate opens a three.js replay for 1d, 2d and 3d models; the scrubber moves the bodies; Esc closes it."""
import asyncio, os
from common import main, check, load
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "..", "examples")
SHOT = os.environ.get("REPLAY_SHOTS", "")
CASES = [("learn/1d_1_mass_spring_damper.yaml", "1d"), ("learn/2d_3_four_bar.yaml", "2d"), ("learn/3d_3_robot_arm.yaml", "3d"),
         ("case_studies/combat_robot/flapjack_drive_3d.yaml", "3d")]

async def s(pg):
    ok = True
    for rel, dim in CASES:
        await load(pg, open(os.path.join(EX, rel)).read())
        await pg.evaluate("() => { document.getElementById('bLive').checked = false; }")
        await pg.evaluate("() => animOpen()")
        for _ in range(600):
            await pg.wait_for_timeout(200)
            if await pg.evaluate("() => AV.open && !!AV.S"):
                break
        st = await pg.evaluate("() => [AV.open, AV.S && AV.S.dim, AV.S && AV.S.t.length, !!AV.R, document.getElementById('smsg').textContent]")
        ok &= check(st[0] and st[1] == dim and st[2] > 100 and st[3], st, f"{dim} replay open", rel)
        # scrub to the middle and to the end: a moving body is somewhere else
        p0 = await pg.evaluate("() => { const r = AV.bodies.find(r => r.b.pos && !r.b.ground); return r.g.position.toArray(); }")
        await pg.evaluate("() => { const e = document.getElementById('avScrub'); e.value = '1000'; e.oninput({target:e}); }")
        p1 = await pg.evaluate("() => { const r = AV.bodies.find(r => r.b.pos && !r.b.ground); return [r.g.position.toArray(), r.g.quaternion.toArray()]; }")
        moved = sum(abs(a - b) for a, b in zip(p0, p1[0])) > 0 or abs(p1[1][3]) < 0.9999
        ok &= check(moved, (p0, p1), "the scrubber moves the bodies", "scrub")
        await pg.evaluate("() => { const e = document.getElementById('avScrub'); e.value = '500'; e.oninput({target:e}); }")
        await pg.wait_for_timeout(300)
        if SHOT:
            await pg.screenshot(path=os.path.join(SHOT, f"replay_{os.path.basename(rel)[:-5]}.png"))
        await pg.evaluate("() => avPlay()"); await pg.wait_for_timeout(500); await pg.evaluate("() => avPlay()")
        await pg.keyboard.press("Escape"); await pg.wait_for_timeout(200)
        cl = await pg.evaluate("() => [AV.open, document.getElementById('animView').style.display]")
        ok &= check(cl == [False, "none"], cl, "Esc closes it", "close")
    return ok

asyncio.run(main(s))

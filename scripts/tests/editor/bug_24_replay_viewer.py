"""Animate opens a three.js replay for 1d, 2d and 3d models; the scrubber moves the bodies and the readout; Esc closes it.
A circuit-only model (no moving bodies) says so and still reads its signals out."""
import asyncio, os
from common import main, check, load
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "..", "examples")
SHOT = os.environ.get("REPLAY_SHOTS", "")
CASES = [("learn/1d_1_mass_spring_damper.yaml", "1d"), ("learn/2d_3_four_bar.yaml", "2d"), ("learn/3d_3_robot_arm.yaml", "3d"),
         ("case_studies/combat_robot/flapjack_drive_3d.yaml", "3d"), ("learn/1d_6_chua_circuit.yaml", "1d"),
         ("learn/3d_1_spinning_top.yaml", "3d")]

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
        rd0 = await pg.evaluate("() => (AV.read || []).map(r => r.el.textContent)")
        ok &= check(len(rd0) > 0, rd0, "a readout is shown", "readout")
        if "chua" in rel:
            em = await pg.evaluate("() => [AV.S.empty, document.getElementById('avEmpty').hidden]")
            ok &= check(em == [True, False], em, "says there are no moving bodies", "empty")
            await pg.evaluate("() => { const e = document.getElementById('avScrub'); e.value = '700'; e.oninput({target:e}); }")
            rd1 = await pg.evaluate("() => (AV.read || []).map(r => r.el.textContent)")
            ok &= check(rd1 != rd0, (rd0, rd1), "the readout follows the scrubber", "readout-moves")
            if SHOT:
                await pg.screenshot(path=os.path.join(SHOT, "replay_chua.png"))
            await pg.keyboard.press("Escape"); await pg.wait_for_timeout(200)
            continue
        if "spinning_top" in rel:                # a body's shape is drawn (the top's disc as a cylinder)
            cy = await pg.evaluate("() => AV.S.bodies.flatMap(b => b.prims.map(p => p.kind))")
            ok &= check("cylinder" in cy, cy, "the top's disc is drawn", "shape")
        if dim != "3d":                          # flat views look straight down and a drag pans, never tilts
            el0 = await pg.evaluate("() => [AV.orbit.el, AV.orbit.az]")
            box = await pg.evaluate("() => { const r = document.getElementById('avCanvas').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }")
            await pg.mouse.move(box[0], box[1]); await pg.mouse.down(); await pg.mouse.move(box[0] + 120, box[1] + 60, steps=6); await pg.mouse.up()
            el1 = await pg.evaluate("() => [AV.orbit.el, AV.orbit.az]")
            ok &= check(abs(el0[0] - 1.5707) < 1e-3 and el0 == el1, (el0, el1), "top-down, drag does not rotate", "flat-lock")
        r0 = await pg.evaluate("() => AV.orbit.r")
        await pg.evaluate("() => { AV.orbit.r *= 7; avCamera(); }")
        await pg.keyboard.press("f")
        r1 = await pg.evaluate("() => AV.orbit.r")
        ok &= check(abs(r1 - r0) < 1e-6 * max(1, r0), (r0, r1), "F fits it again", "fit-key")
        if "flapjack" in rel:                    # the hull is a sketch that can be switched off
            hv = await pg.evaluate("() => { const vis = () => AV.bodies.flatMap(r => r.g.children.filter(o => o.userData.hull).map(o => o.visible));"
                                   " const a = vis(); const c = document.getElementById('avHull'); c.checked = false; c.onchange(); const b = vis();"
                                   " c.checked = true; c.onchange(); return [a, b]; }")
            ok &= check(len(hv[0]) > 0 and all(hv[0]) and not any(hv[1]), hv, "hulls on, then off", "hull-toggle")
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

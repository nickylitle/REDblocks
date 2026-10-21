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
        if "flapjack" in rel:                    # beams solid or thin: no beams here, the box is there all the same
            ok &= check(await pg.evaluate("() => !!document.getElementById('avBeams')"), True, "beams toggle", "beams-toggle")
        if "spinning_top" in rel:                # a body's shape is drawn (the top's disc as a cylinder)
            cy = await pg.evaluate("() => AV.S.bodies.flatMap(b => b.prims.map(p => p.kind))")
            ok &= check("cylinder" in cy, cy, "the top's disc is drawn", "shape")
        cam = await pg.evaluate("() => !!AV.cam.isOrthographicCamera")
        ok &= check(cam == (dim != "3d"), cam, "1D and 2D drawn flat (no perspective), 3D in perspective", "flat-style")
        mg = await pg.evaluate("() => { const box = document.getElementById('avMag'); if (!AV.S.deform) return ['none', box.disabled];"
                               " const t = AV.S.deform.find(x => x.el); const r = AV.bodies[t.b]; AV.tcur = AV.S.t[Math.floor(AV.S.t.length / 3)]; avPose(); const a = r.g.position.clone();"
                               " const p0 = avPosesAt(0, 0)[t.b][0]; box.value = String((AV.S.mag_auto || 1) * 3); box.oninput();"
                               " const p03 = avPosesAt(0, 0)[t.b][0]; avPose(); const b = r.g.position.clone(); box.value = ''; box.oninput(); AV.tcur = AV.S.t[0]; avPose();"
                               " return [a.x - p0.x, b.x - p03.x, a.y - p0.y, b.y - p03.y]; }")
        if mg[0] == "none":
            ok &= check(mg[1] is True, mg, "nothing deforms: the box is off", "motion-mult")
        else:
            ok &= check(abs(mg[0]) + abs(mg[2]) > 0 and all(abs(mg[i + 1] - 3 * mg[i]) < 0.02 * abs(mg[i]) + 1e-9 for i in (0, 2)), mg, "deform x 3 triples a spring's stretch (smoothed over 3 frames)", "motion-mult")
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
        if dim == "3d" and "robot_arm" in rel:   # the whole run in a chosen number of seconds
            sp = await pg.evaluate("() => { const T = AV.T; const b = document.getElementById('avFitS'); b.value = '25'; b.oninput();"
                                   " const r25 = AV.rate; const s = document.getElementById('avSpeed'); s.value = '5'; s.onchange();"
                                   " const r5 = [AV.rate, b.hidden]; s.value = 'fit'; s.onchange(); return [T, r25, r5]; }")
            ok &= check(abs(sp[1] - sp[0] / 25) < 1e-9 and sp[2] == [5, True], sp, "fit to 25 s; 5x hides the seconds box", "speed")
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

"""Spam-clicking Run starts one run and throws confetti once; while it runs the button is Stop, and Stop ends it."""
import asyncio, os
from common import main, check, load
HERE = os.path.dirname(os.path.abspath(__file__))
LONG = open(os.path.join(HERE, "..", "..", "examples", "learn", "3d_3_robot_arm.yaml")).read().replace("t_end: 2.5 s", "t_end: 40 s")   # long enough to stop
SHORT = open(os.path.join(HERE, "..", "..", "examples", "learn", "2d_1_pendulum.yaml")).read()

async def spam(pg, n=6):
    for _ in range(n):
        await pg.evaluate("() => document.getElementById('bRun').click()")
        await pg.wait_for_timeout(40)

async def s(pg):
    await pg.evaluate("() => { window.__conf = 0; const c = confetti; confetti = function(){ window.__conf++; return c.apply(this, arguments); }; }")
    # a short run, Run spammed: one run, one confetti
    await load(pg, SHORT)
    await pg.evaluate("() => { document.getElementById('bLive').checked = false; }")
    await spam(pg)
    for _ in range(100):
        await pg.wait_for_timeout(100)
        if not await pg.evaluate("() => runBusy"):
            break
    await pg.wait_for_timeout(1200)
    n = await pg.evaluate("() => window.__conf")
    ok = check(n == 1, n, "confetti once for a spammed Run", "confetti")
    # a long run: the button turns into Stop, a double-click does not stop it, a later click does
    await load(pg, LONG)
    await pg.evaluate("() => { window.__conf = 0; }")
    await spam(pg, 2)
    st = await pg.evaluate("() => [runBusy, document.getElementById('bRun').textContent]")
    ok &= check(st == [True, "Stop"], st, "running, button reads Stop", "stop-label")
    await pg.wait_for_timeout(1500)
    st = await pg.evaluate("() => runBusy")
    ok &= check(st, st, "the double-click did not stop it", "no-double")
    await pg.evaluate("() => document.getElementById('bRun').click()")
    for _ in range(100):
        await pg.wait_for_timeout(100)
        if not await pg.evaluate("() => runBusy"):
            break
    msg = await pg.evaluate("() => [document.getElementById('bRun').textContent, document.getElementById('status').textContent, window.__conf]")
    ok &= check(msg[0] == "Run" and "Stopped at" in msg[1] and msg[2] == 0, msg, "back to Run, partial results, no confetti", "stopped")
    await pg.wait_for_timeout(600)
    await pg.evaluate("() => document.getElementById('bRun').click()")       # the next run is not cancelled
    for _ in range(200):
        await pg.wait_for_timeout(100)
        if not await pg.evaluate("() => runBusy"):
            break
    msg = await pg.evaluate("() => [document.getElementById('status').textContent, window.__conf]")
    ok &= check(msg[0].startswith("Ran in") and msg[1] == 1, msg, "a fresh Run after a Stop runs to the end", "rerun")
    return ok

asyncio.run(main(s))

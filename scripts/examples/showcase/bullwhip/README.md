# Bullwhip crack

A 2.5 m tapered bullwhip, thrown overhead. The arm swings from behind to forward in 0.14 s and stops; a loop runs
down the whip, and because the whip thins (16 mm to 1.8 mm) the same energy rides on less and less mass, so the loop
speeds up. The tip passes the speed of sound at about 134 ms: the crack is the small sonic boom it makes.

| | in air | vacuum (scenario) |
|---|---|---|
| fastest tip speed | about 479 m/s (Mach 1.40) | about 625 m/s (Mach 1.8) |

What is in it:
- 12 beams, each its own round section, split finer toward the tip (58 segments): the loop is tightest there.
- Air drag on every node: crossflow drag on the speed across the whip, with a Cd that rises through Mach 1
  (Hoerner), and the cracker's own drag. `air: 0` (scenario `vacuum`) turns it off.
- The shoulder: a rigid hinge, driven by a stiff PD torque following a smooth throw.

Checks (tests/test_showcase_bullwhip.py): the drag law against terminal speed and coasting closed forms, the energy
ledger closes to 0.01 %, a finer mesh moves the tip speed by about 1.5 %, every segment bends under 30 deg over its
chord. Real whips crack at about Mach 1 to 2 (Krehl et al. 1998, McMillen and Goriely 2003).

In vacuum the loop keeps tightening (a whip without air focuses its energy without limit), so that run stops just
after the crack, where the mesh still follows it.

`build_whip.py` writes the model: change the length, taper or mesh there.

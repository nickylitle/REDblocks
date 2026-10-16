"""Full verification battery on the block-built wedge strike (legacy physics)."""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedge_strike_from_blocks import wedge_strike, critical_dt
from simblocks.diagnostics.verify import SimSpec, verify_all, plot_report

LEG = dict(mat_overrides=dict(E=2.1e11, nu=0.3))
spec = SimSpec(
    name="horizontal wedge strike (legacy physics, block-built)",
    run=lambda p, dt: wedge_strike(p, dt),
    params=dict(LEG, t=0.002), dt=1e-7,
    metrics=["bending deformation", "denting deformation", "wedge drop", "weapon_energy_lost"],
    uncertain={"t": (0.0015, 0.0025), "weapon_angle0_deg": (-40.0, -20.0),
               "weapon_w": (1130.0, 1380.0), "rate_factor": (1.1, 1.3)},
    knockouts={"no_dent_plasticity": (dict(dent_on=0.0), {"denting deformation": "-"}),
               "no_bend_plasticity": (dict(bend_on=0.0), {"bending deformation": "-"})},
    limits=[("weapon stopped -> no damage", dict(weapon_w=1e-9),
             {"bending deformation": 0.0, "denting deformation": 0.0}),
            ("rigid wedge -> no permanent set", dict(rate_factor=1e6),
             {"bending deformation": 0.0, "denting deformation": 0.0})],
    bounds={"bending deformation": (0.0, 90.0), "denting deformation": (0.0, 30.0), "wedge drop": (-30.0, 30.0)},
    critical_dt=critical_dt, mc_n=200, conv_tol=0.01)
rep = verify_all(spec)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(out, exist_ok=True)
open(os.path.join(out, "wedge_verification.md"), "w").write(rep.markdown())
plot_report(rep, os.path.join(out, "wedge_verification.png"))
print(rep.markdown())

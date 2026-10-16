import sys, os, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedge_strike_from_blocks import wedge_strike, critical_dt, build, DEFAULTS
from simblocks.diagnostics.verify import SimSpec, verify_all, plot_report
LEG = dict(mat_overrides=dict(E=2.1e11, nu=0.3))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

print("== thickness sweep, nominal hit (weapon starts -30 deg, bot at rest) ==")
print("t mm | legacy bend deg, dent mm | curve kappa=1 bend, dent | curve kappa=2.2 bend, dent")
for t in (0.0015, 0.002, 0.0025, 0.003):
    a = wedge_strike(dict(LEG, t=t)); b = wedge_strike(dict(t=t, physics="curve"))
    c = wedge_strike(dict(t=t, physics="curve", edge_clamp_factor=2.2))
    print(f"{t*1e3:.1f} | {a['bending deformation']:6.2f} {a['denting deformation']:5.2f} | "
          f"{b['bending deformation']:6.2f} {b['denting deformation']:5.2f} | {c['bending deformation']:6.2f} {c['denting deformation']:5.2f}")

def spec(kappa):
    return SimSpec(
        name=f"wedge 2.5 mm curve physics kappa={kappa}",
        run=lambda p, dt: wedge_strike(p, dt),
        params=dict(t=0.0025, physics="curve", edge_clamp_factor=kappa), dt=1e-7,
        metrics=["bending deformation", "denting deformation", "wedge drop", "weapon_energy_lost"],
        uncertain={"weapon_angle0_deg": (-40.0, -20.0), "weapon_w": (1130.0, 1380.0),
                   "rate_factor": (1.1, 1.3), "hinge_len_over_t": (0.5, 2.0), "edge_clamp_factor": (0.9 * kappa, 1.1 * kappa)},
        knockouts={"no_dent_plasticity": (dict(dent_on=0.0), {"denting deformation": "-"}),
                   "no_bend_plasticity": (dict(bend_on=0.0), {"bending deformation": "-"})},
        limits=[("weapon stopped -> no damage", dict(weapon_w=1e-9), {"bending deformation": 0.0, "denting deformation": 0.0}),
                ("rigid wedge -> no permanent set", dict(rate_factor=1e6, edge_clamp_factor=1.0),
                 {"bending deformation": lambda v: abs(v) < 1e-9, "denting deformation": lambda v: abs(v) < 1e-9})],
        bounds={"bending deformation": (0.0, 90.0), "denting deformation": (0.0, 30.0)},
        critical_dt=critical_dt, mc_n=300, conv_tol=0.01)

for kappa in (1.0, 2.2):
    rep = verify_all(spec(kappa))
    tag = f"k{kappa:.1f}".replace(".", "p")
    open(os.path.join(OUT, f"wedge_2p5mm_curve_{tag}.md"), "w").write(rep.markdown())
    plot_report(rep, os.path.join(OUT, f"wedge_2p5mm_curve_{tag}.png"))
    print("\n" + "\n".join(rep.markdown().splitlines()[:30]))
    mc = next(c for c in rep.checks if c.name == "monte_carlo")
    Y = np.array(mc.data["Y"]); print("MC max bend", Y[:, 0].max(), "max dent", Y[:, 1].max())

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedge_strike_from_blocks import wedge_strike, critical_dt
from simblocks.geometry.contact_geometry import first_contact_delta, first_contact_stats, sample_first_contact
from simblocks.diagnostics.verify import SimSpec, verify_all, plot_report
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rpm = 1257 * 60 / (2 * math.pi); D = 0.18
d = first_contact_delta(3.0, rpm, D); st = first_contact_stats(d)
amax, amed = math.degrees(st["alpha_max"]), math.degrees(st["median"])
print(f"delta {d:.3f}  alpha_max {amax:.1f} deg  median {amed:.1f} deg (3 m/s closing, {rpm:.0f} rpm, {D*1e3:.0f} mm)")
for kappa in (1.0, 2.2):
    for lab, a in (("median", amed), ("worst alpha_max", amax)):
        r = wedge_strike(dict(t=0.0025, physics="curve", edge_clamp_factor=kappa, weapon_angle0_deg=-a))
        print(f"kappa {kappa} {lab:16s}: bend {r['bending deformation']:5.2f} deg, dent {r['denting deformation']:.2f} mm, "
              f"weapon energy into hit {r['weapon_energy_lost']:.0f} J")
    sp = SimSpec(name=f"wedge 2.5 mm, first-contact distribution, kappa {kappa}", run=lambda p, dt: wedge_strike(p, dt),
                 params=dict(t=0.0025, physics="curve", edge_clamp_factor=kappa, weapon_angle0_deg=-amed), dt=1e-7,
                 metrics=["bending deformation", "denting deformation", "weapon_energy_lost"],
                 uncertain={"weapon_angle0_deg": lambda rng: -math.degrees(sample_first_contact(d, 1, rng)[0]),
                            "weapon_w": (1130.0, 1380.0), "rate_factor": (1.1, 1.3), "hinge_len_over_t": (0.5, 2.0),
                            "edge_clamp_factor": (0.9 * kappa, 1.1 * kappa)},
                 knockouts={"no_dent_plasticity": (dict(dent_on=0.0), {"denting deformation": "-"}),
                            "no_bend_plasticity": (dict(bend_on=0.0), {"bending deformation": "-"})},
                 limits=[("weapon stopped -> no damage", dict(weapon_w=1e-9), {"bending deformation": 0.0})],
                 bounds={"bending deformation": (0.0, 90.0), "denting deformation": (0.0, 30.0)},
                 critical_dt=critical_dt, mc_n=400)
    rep = verify_all(sp, skip=("sensitivity",))
    tag = f"k{kappa:.1f}".replace(".", "p")
    open(os.path.join(OUT, f"wedge_2p5mm_firstcontact_{tag}.md"), "w").write(rep.markdown())
    plot_report(rep, os.path.join(OUT, f"wedge_2p5mm_firstcontact_{tag}.png"))
    mc = next(c for c in rep.checks if c.name == "monte_carlo")
    print("  ", rep.summary())
    for m, q in mc.data["percentiles"].items():
        print(f"   {m:22s} P5 {q[0]:.2f}  P50 {q[1]:.2f}  P95 {q[2]:.2f}   max {np.max(np.array(mc.data['Y'])[:, mc.data['metrics'].index(m)]):.2f}")

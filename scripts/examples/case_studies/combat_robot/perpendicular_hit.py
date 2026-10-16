"""What if the vertical blade's tooth is moving perpendicular to Flapjack's blade (straight up) at first contact?

The tooth moves straight up only at the front of its circle, at axle height. So the case is set up by putting
Flapjack's blade at the vertical spinner's axle height (H_zblade = V_R + V_clear), same 160 engagements as the
baseline tip-load study, same blade strength (flap collapse 6110 N). Compares Z impulse and peak with the baseline.
"""
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from spinner_vs_spinner import run, DEFAULTS
from rigid_blade_tip_load import strength_scale, _one, HIST_DT, tip_spec_run

OUT = os.path.join(HERE, "out", "perpendicular_hit"); os.makedirs(OUT, exist_ok=True)


def main():
    scale, _ = strength_scale()
    mc = pickle.load(open(os.path.join(HERE, "out", "spinner_vs_spinner", "mc.pkl"), "rb"))
    z_ax = DEFAULTS["V_R"] + DEFAULTS["V_clear"]
    base = dict(H_sy_scale=scale, hist_dt=HIST_DT, H_zblade=z_ax)
    cache = os.path.join(OUT, "runs.pkl")
    if os.path.exists(cache) and "--rerun" not in sys.argv:
        R = pickle.load(open(cache, "rb"))
    else:
        from concurrent.futures import ProcessPoolExecutor
        t0 = time.time(); run(dict(t_end=2e-4))
        with ProcessPoolExecutor() as ex:
            R = list(ex.map(_one, [(i, {**s, **base}) for i, s in enumerate(mc["samples"])], chunksize=2))
        pickle.dump(R, open(cache, "wb")); print(f"{len(R)} runs in {time.time()-t0:.0f} s")
    B = pickle.load(open(os.path.join(HERE, "out", "rigid_blade_tip_load", "runs.pkl"), "rb"))["R"]
    for name, S in (("baseline (blade at 15 mm)", B), (f"perpendicular (blade at {z_ax*1e3:.0f} mm)", R)):
        jz = np.array([r["z"]["J_z"] for r in S]); fz = np.array([r["z"]["Fz_pk"] for r in S])
        hit = np.array([abs(r["z"]["J_z"]) > 0.05 for r in S])
        jy = np.array([r["z"]["J_y"] for r in S])
        ang = np.degrees(np.arctan2(np.abs(jz), np.abs(jy)))
        print(f"{name}: blade hit {hit.mean()*100:.0f}% | Z impulse max {jz.max():+.2f}, min {jz.min():+.2f}, "
              f"median|hit {np.median(jz[hit]):+.2f} N s | Fz peak max {fz.max():.0f} N | "
              f"impulse angle above blade plane median {np.median(ang[hit]):.0f} deg")
    top = sorted(R, key=lambda r: abs(r["z"]["J_z"]), reverse=True)[:4]
    for r in top:
        lv = [tip_spec_run(r["params"], d) for d in (2.5e-7, 1.25e-7)]
        print(f"sample {r['i']}: J_z {r['z']['J_z']:+.3f} (dt/2 {lv[1]['tip_J_z']:+.3f}), Fz pk {r['z']['Fz_pk']:.0f} "
              f"(dt/2 {lv[1]['tip_Fz_pk']:.0f}), J_y {r['z']['J_y']:+.3f}, pulses {r['z']['n_pulses']}")
    return R


if __name__ == "__main__":
    main()

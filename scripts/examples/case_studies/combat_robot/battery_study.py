"""Monte Carlo of the minimum 6S pack for a 180 s fight (battery_life.py). Sobol samples over fight use and
hardware uncertainty; for each sample the smallest rated capacity that passes is found by bisection, under three
cutoffs: capacity only (80 % of rated mAh), loaded voltage >= 3.3 V/cell, loaded voltage >= 3.0 V/cell."""
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from battery_life import run, min_capacity
from simblocks.batch import make_samples

OUT = os.path.join(HERE, "out", "battery_life"); os.makedirs(OUT, exist_ok=True)
N = 512
DIMS = {  # name: (lo, hi, label, unit)
    "n_spinups": (3, 20, "weapon spin-ups per fight", ""),
    "f_after_hit": (0.0, 0.8, "blade speed left after a hit", "x target"),
    "push_s": (0.0, 60.0, "pushing time", "s"),
    "duty_drive": (0.3, 0.8, "normal driving, share of the rest", ""),
    "I_lim_w": (25.0, 45.0, "weapon ESC current limit, per motor", "A"),
    "r450_mohm": (8.0, 25.0, "cell resistance (450 mAh cell)", "mOhm"),
    "Cd_wind": (0.05, 0.4, "blade windage Cd (edge)", ""),
    "I0_w": (1.0, 2.0, "weapon motor no-load current", "A"),
    "T_bearing": (0.002, 0.02, "hub + tread drag torque", "N m"),
    "eta_fd": (0.75, 0.95, "friction drive efficiency", ""),
    "G_w": (1.25, 1.5, "friction drive ratio (motor:blade)", ""),
    "J_blade": (7.6e-4, 8.8e-4, "blade inertia", "kg m^2"),
    "F_push": (8.0, 150.0, "push traction limit, both wheels", "N"),
    "R_w": (0.045, 0.058, "weapon motor resistance", "Ohm"),
}
CUTS = {"capacity only": 0.0, "3.3 V/cell": 3.3, "3.0 V/cell": 3.0}


def one(a):
    i, s = a
    s = dict(s); s["n_spinups"] = int(round(s["n_spinups"])); s["seed"] = i
    rec = dict(i=i, params=s)
    for name, vc in CUTS.items():
        c, o = min_capacity({**s, "v_cut": vc}, lo=150.0, hi=2500.0, tol=10.0)
        rec[name] = dict(cap=c, governs=(o or {}).get("governs"), Wh=(o or {}).get("Wh"), I_peak=(o or {}).get("I_peak"),
                         v_min_cell=(o or {}).get("v_min_cell"), C_peak=(o or {}).get("C_peak"),
                         ledger=(o or {}).get("_ledger_rel"))
    o650 = run({**s, "cap_mAh": 650.0}); o450 = run({**s, "cap_mAh": 450.0})
    rec["at450"] = {k: o450[k] for k in ("Wh", "used_frac", "v_min_cell", "I_peak", "passes", "_ledger_rel")}
    rec["at650"] = {k: o650[k] for k in ("Wh", "used_frac", "v_min_cell", "I_peak", "passes", "_ledger_rel")}
    rec["E"] = {k: o650[k] for k in o650 if k.startswith("E_") or k == "dKE_blade"}
    return rec


def main():
    cache = os.path.join(OUT, "mc.pkl")
    if os.path.exists(cache) and "--rerun" not in sys.argv:
        return pickle.load(open(cache, "rb"))
    samples = make_samples(dict(samples=dict(n=N, sampler="sobol", seed=3,
                                           uniform={k: (v[0], v[1]) for k, v in DIMS.items()})))
    run({})
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        R = list(ex.map(one, list(enumerate(samples)), chunksize=4))
    print(f"{len(R)} samples in {time.time()-t0:.0f} s")
    mc = dict(R=R, dims=DIMS, cuts=CUTS, n=N)
    pickle.dump(mc, open(cache, "wb"))
    return mc


if __name__ == "__main__":
    mc = main()
    R = mc["R"]
    for c in CUTS:
        caps = np.array([r[c]["cap"] for r in R])
        print(c, "min mAh percentiles 5/50/95/max:", np.nanpercentile(caps, [5, 50, 95, 100]).round(0),
              "governs:", {g: sum(1 for r in R if r[c]["governs"] == g) for g in ("capacity", "sag")},
              "nan", int(np.isnan(caps).sum()))
    print("worst ledger", max(r[c]["ledger"] or 0 for r in R for c in CUTS))

"""Run the ORIGINAL Horizontal Wedge Impact.py simulate_strike and the block rebuild side by side."""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedge_strike_from_blocks import wedge_strike

src = open(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Horizontal_Wedge_Impact.py").read()
src = src[:src.index("thicknesses =")]          # function only, no plotting
ns = {}; exec(src, ns)
orig = ns["simulate_strike"]

worst = {}
for t in (0.001, 0.0015, 0.002, 0.0025, 0.003):
    t0 = time.time(); a = orig(t, np.radians(-30), 0); ta = time.time() - t0
    t0 = time.time(); b = wedge_strike(dict(t=t, mat_overrides=dict(E=2.1e11, nu=0.3))); tb = time.time() - t0
    row = []
    for k in ("bending deformation", "denting deformation", "wedge drop", "force"):
        d = abs(b[k] - a[k]) / max(abs(a[k]), 1e-30)
        worst[k] = max(worst.get(k, 0), d)
        row.append(f"{k[:5]} {a[k]:.6g} vs {b[k]:.6g}")
    print(f"t={t*1e3:.1f} mm | " + " | ".join(row) + f" | {ta:.1f}s vs {tb:.2f}s | ledger {b['_ledger_rel']*100:.2f}%")
print("worst relative difference:", {k: f"{v:.2e}" for k, v in worst.items()})

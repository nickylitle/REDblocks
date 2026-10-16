"""Process diagram of the rigid-blade tip-load calculation: sim -> measured arm force -> window -> rank -> verify ->
export -> gyro moment -> the two Ansys tables. Two rows, read left to right. Numbers match the delivered tables."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from simblocks.viz import theme as T

OUT = os.path.join(HERE, "out", "rigid_blade_tip_load")

STAGES = [
    ("1  Hit sims", "state", [("spinner_vs_spinner", "Flapjack vs vertical spinner"),
                              ("H blade 30 x 6.35 AR500", "R 95 mm, 11360 rpm"),
                              ("arm yields at 6110 N", "full plastic collapse + rate")]),
    ("2  Measure", "force", [("arm force on the blade", "at the tip, every step"),
                             ("arm frame", "radial / lag / flap (Z)"),
                             ("radial - m R w^2", "centrifugal part removed")]),
    ("3  Window", "compute", [("contact + unloading", "from first blade contact"),
                              ("each axis: own zero", "crossing after last contact"),
                              ("radial dropped", "85-148 kN peaks unreliable")]),
    ("4  Rank", "constitutive", [("signed Z impulse", "not the |F| integral"),
                                 ("top 8 candidates", "worst first"),
                                 ("yours: 2.30 N s", "two 3408 N half-sines")]),
    ("5  Verify", "integrate", [("re-run at smaller dt", "Richardson change < 1%"),
                                ("fails: next candidate", "loops back to rank"),
                                ("worst Z: 2.81 N s", "peak 6109 N, 1.22x yours")]),
    ("6  Export", "output", [("fe_load_table", "simplified, impulse exact"),
                             ("Fz only", "in-plane dropped (overbuilt)"),
                             ("tip -> blade centre", "Mx = 0.095 m x Fz")]),
    ("7  Gyro moment", "geometry", [("Flapjack alone on wheels", "blade spinning, load at tip"),
                                    ("M = -w x h", "h = 1.01 N m s"),
                                    ("My peak 77 N m", "Mx peak 578 N m")]),
    ("Ansys tables", "result", [("force table", "t, 0, 0, Fz   17 rows"),
                                ("moment table", "t, Mx, My, 0   22 rows"),
                                ("apply at blade centre", "run to 5 ms")]),
]
EDGES = ["history", "F(t) xyz", "pulses", "candidate", "worst Z", "Fz(t)", "Mx, My"]
ROLE = {"state": T.ACCENT, "force": T.SERIES[4], "compute": T.EDGE, "constitutive": T.SERIES[1],
        "integrate": T.SERIES[2], "output": T.SERIES[3], "geometry": T.SERIES[0], "result": T.STATUS["PASS"]}


def main():
    T.apply()
    fig = plt.figure(figsize=(15.0, 8.4))
    top = T.header(fig, "How the tip-load tables were calculated",
                   "Flexible-blade hit sims -> force the arm puts on a rigid blade -> worst verified Z hit -> "
                   "force and moment tables at the blade centre", tag="SIMBLOCKS // LOAD EXPORT PIPELINE")
    ax = fig.add_axes([0.03, 0.05, 0.94, top - 0.08]); ax.axis("off")
    W, H, gx, gy = 3.0, 2.55, 0.95, 1.25
    pos = []
    for i, (title, role, items) in enumerate(STAGES):
        r, c = divmod(i, 4)
        x, y = c * (W + gx), -r * (H + gy)
        pos.append((x, y))
        hi = role == "result"
        ax.add_patch(FancyBboxPatch((x, y - H), W, H, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=T.PANEL, ec=ROLE[role] if hi else T.EDGE, lw=1.4 if hi else 1.0, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y - 0.5), W, 0.5, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=T.PANEL_2, ec=ROLE[role] if hi else T.EDGE, lw=1.0, zorder=3))
        ax.add_patch(plt.Rectangle((x, y - 0.5), 0.07, 0.5, color=ROLE[role], zorder=4, lw=0))
        ax.text(x + 0.2, y - 0.25, title.upper(), va="center", fontsize=9, color=T.INK, family=T.MONO,
                weight="bold", zorder=5)
        for j, (name, note) in enumerate(items):
            yy = y - 0.85 - 0.6 * j
            ax.text(x + 0.2, yy, name, fontsize=8.6, color=T.INK, family=T.MONO, va="center", zorder=5)
            ax.text(x + 0.2, yy - 0.25, note, fontsize=7.8, color=T.INK_3, va="center", zorder=5)

    def arrow(p0, p1, col=T.INK_2):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=col, lw=1.2, zorder=6))

    for i, lab in enumerate(EDGES):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            ym = y0 - H / 2
            arrow((x0 + W + 0.06, ym), (x1 - 0.06, ym))
            ax.text((x0 + W + x1) / 2, ym + 0.12, lab, ha="center", va="bottom", fontsize=7.6, color=T.INK_3,
                    family=T.MONO)
        else:                                  # row wrap: down from stage 4, back left, into stage 5
            xa, xb, yc = x0 + W / 2, x1 + W / 2, y0 - H - gy / 2
            ax.plot([xa, xa, xb], [y0 - H - 0.04, yc, yc], color=T.INK_2, lw=1.2, zorder=6)
            arrow((xb, yc), (xb, y1 + 0.06))
            ax.text((xa + xb) / 2, yc + 0.1, lab, ha="center", va="bottom", fontsize=7.6, color=T.INK_3,
                    family=T.MONO)
    ax.set_xlim(-0.2, 4 * W + 3 * gx + 0.2); ax.set_ylim(-2 * H - gy - 0.2, 0.25); ax.set_aspect("equal")
    T.footer(fig, "Stage 5 loops back to stage 4 when a candidate's peak changes more than 1% at smaller dt. "
                  "Frame: X side to side, Y forward/back, Z up, blade pointing forward.")
    return T.save(fig, os.path.join(OUT, "tip_load_calc_flow"))


if __name__ == "__main__":
    print(main())

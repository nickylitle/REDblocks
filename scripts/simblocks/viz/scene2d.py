"""Drawing of a 2d model: every body's points, circles and lines at the start (solid) and, if given, at a later
pose (ghost), plus what each element joins. Drawn from the compiled model, so it shows exactly what the solver uses."""
import math

import numpy as np

from . import theme as T


def _pt(q, i, x, y):
    if i < 0:
        return x, y
    c, s = math.cos(q[3 * i + 2]), math.sin(q[3 * i + 2])
    return q[3 * i] + c * x - s * y, q[3 * i + 1] + s * x + c * y


def _draw_pose(ax, M, q, alpha, lw, label=True):
    names = M.body_names
    for (b, fn), (kind, g) in M.feats.items():
        i = names.index(b) if b in names else -1
        col = T.EDGE if i < 0 else T.SERIES[i % len(T.SERIES)]
        if fn == "__origin__":
            continue
        if kind == "line":
            (x0, y0), (x1, y1) = _pt(q, i, g[0], g[1]), _pt(q, i, g[2], g[3])
            ax.plot([x0, x1], [y0, y1], color=col, lw=lw * 1.6, alpha=alpha, solid_capstyle="round")
            if i < 0:                                            # hatch the solid side of ground lines
                L = math.hypot(x1 - x0, y1 - y0); tx, ty = (x1 - x0) / L, (y1 - y0) / L
                for u in np.linspace(0, L, 14)[1:]:
                    px, py = x0 + tx * u, y0 + ty * u
                    ax.plot([px, px - 0.04 * L * (tx + ty)], [py, py - 0.04 * L * (ty - tx)], color=col, lw=0.6, alpha=alpha)
            if label:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, f" {b}.{fn}", color=col, fontsize=6.5, family=T.MONO, alpha=alpha)
        else:
            cx, cy = _pt(q, i, g[0], g[1]); r = g[2]
            if r > 0:
                ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                    (cx, cy), r, fill=False, color=col, lw=lw, alpha=alpha))
            else:
                ax.plot([cx], [cy], "o", ms=3.5, color=col, alpha=alpha)
            if label:
                ax.text(cx, cy, f" {b}.{fn}", color=col, fontsize=6.5, family=T.MONO, alpha=alpha, va="bottom")
    for i, b in enumerate(names):
        if label and M.b2[b]["spin_only"]:                     # swing circle of each point on a spinning body
            import matplotlib.patches as mp_
            for (bb_, fn), (kind, g) in M.feats.items():
                if bb_ == b and kind == "circle" and fn != "__origin__":
                    rr = math.hypot(g[0], g[1]) + g[2]
                    if rr > 0:
                        ax.add_patch(mp_.Circle((q[3 * i], q[3 * i + 1]), rr, fill=False, ls=(0, (2, 3)), lw=0.7,
                                                color=T.SERIES[i % len(T.SERIES)], alpha=0.6))
    for i, b in enumerate(names):
        x, y = q[3 * i], q[3 * i + 1]
        col = T.SERIES[i % len(T.SERIES)]
        ax.plot([x], [y], marker="+", ms=8, color=col, alpha=alpha, mew=1.2)
        if label:
            ax.text(x, y, f"  {b}", color=col, fontsize=7.5, weight="bold", alpha=alpha, va="top")


def draw(M, path_no_ext, q_end=None, title=None):
    import matplotlib.pyplot as plt
    from ..compiler import engine2d as E2
    T.apply()
    fig = plt.figure(figsize=(8.0, 6.2))
    T.header(fig, title or f"{M.doc.get('name', 'model')}: 2d model", "Solid: start. Faint: end of the run. "
             "+ marks each body's tracked point (centre of mass, or the pivot of a spin-only body)")
    ax = fig.add_axes([0.095, 0.17, 0.875, 0.68]); ax.set_facecolor(T.PANEL)
    a = M.arrays; q0 = a["q0"]
    if q_end is not None:
        _draw_pose(ax, M, q_end, 0.28, 0.9, label=False)
    _draw_pose(ax, M, q0, 1.0, 1.2)
    names = M.body_names; notes = []
    for e, en in enumerate(M.el_names):
        mk = a["mk"][e]; t = M.el_info[en]["type"]
        if mk in (E2.MK_DIST, E2.MK_ALONG, E2.MK_PIN):
            pa = _pt(q0, a["ba"][e], a["GA"][e, 0], a["GA"][e, 1]); pb = _pt(q0, a["bb"][e], a["GB"][e, 0], a["GB"][e, 1])
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=T.ACCENT, lw=1.0, ls=(0, (4, 2)) if mk == E2.MK_ALONG else "-")
            ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, f" {en} ({t})", color=T.ACCENT, fontsize=6.5, family=T.MONO)
        elif mk in (E2.MK_EDGE, E2.MK_CIRC):
            pa = _pt(q0, a["ba"][e], a["GA"][e, 0], a["GA"][e, 1])
            ax.plot([pa[0]], [pa[1]], marker="x", ms=6, color=T.STATUS["WARN"])
            notes.append(f"{en}: {t} on {M.el_info[en]['measure']}")
        else:
            notes.append(f"{en}: {t} on {M.el_info[en]['measure']}")
    ax.set_aspect("equal", adjustable="datalim"); ax.margins(0.12)
    T.style_ticks(ax); ax.grid(color=T.HAIRLINE, lw=0.5)
    ax.set_xlabel("x (m)", fontsize=7, color=T.INK_2 if hasattr(T, "INK_2") else T.INK)
    ax.set_ylabel("y (m)", fontsize=7, color=T.INK_2 if hasattr(T, "INK_2") else T.INK)
    if any(a["grav"]):
        notes.append(f"gravity ({a['grav'][0]:.3g}, {a['grav'][1]:.3g}) m/s^2")
    T.footer(fig, "   ".join(notes)[:400] or "Drawn from the compiled model.", max_lines=3)
    return T.save(fig, path_no_ext, strict=False)

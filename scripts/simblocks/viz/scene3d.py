"""Drawing of a 3d model in three views (top, front, side): every body's points, spheres and plane patches at the
start (solid) and, where the end pose is known (bodies that only translate or spin), at the end (faint)."""
import math

import numpy as np

from . import theme as T

VIEWS = [("top (x, y)", 0, 1), ("side (x, z)", 0, 2), ("front (y, z)", 1, 2)]


def _R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _poses(M, H=None):
    """[(X (nb,3), R (nb,3,3))] for the start and, if possible, the end."""
    from ..compiler.model3d import rot
    a = M.arrays; nb = len(M.body_names)
    R0 = np.array([_R(a["quat0"][b]) for b in range(nb)]) if nb else np.zeros((0, 3, 3))
    out = [(a["X0"].copy(), R0, 1.0)]
    if H is not None and len(H.get("t", [])):
        X1 = np.array([[H[f"{b}.{k}"][-1] for k in ("x", "y", "z")] for b in M.body_names]) if nb else np.zeros((0, 3))
        R1 = R0.copy()
        for i, b in enumerate(M.body_names):
            if a["rmode"][i] == 1:
                R1[i] = rot(a["axis"][i], H[f"{b}.angle"][-1] - a["th0"][i]) @ R0[i]
            elif a["rmode"][i] == 2:
                R1[i] = np.full((3, 3), np.nan)
        out.append((X1, R1, 0.28))
    return out


def draw(M, path_no_ext, H=None, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    T.apply()
    fig = plt.figure(figsize=(11.0, 5.4))
    T.header(fig, title or f"{M.doc.get('name', 'model')}: 3d model",
             "Three views. Solid: start. Faint: end of the run (bodies that translate or spin). + marks each body's "
             "tracked point. z is up.")
    names = M.body_names
    for vi, (vt, ia, ib) in enumerate(VIEWS):
        ax = fig.add_axes([0.06 + vi * 0.32, 0.17, 0.27, 0.62]); ax.set_facecolor(T.PANEL)
        for X, R, alpha in _poses(M, H):
            for (b, fn), (kind, g) in M.feats.items():
                if fn == "__origin__":
                    continue
                i = names.index(b) if b in names else -1
                if i >= 0 and np.isnan(R[i]).any():
                    continue
                col = T.EDGE if i < 0 else T.SERIES[i % len(T.SERIES)]
                tf = (lambda p: np.asarray(p)) if i < 0 else (lambda p, i=i: X[i] + R[i] @ np.asarray(p))
                dr = (lambda v: np.asarray(v)) if i < 0 else (lambda v, i=i: R[i] @ np.asarray(v))
                if kind == "plane":
                    o = tf(g[:3]); n = dr(g[3:6]); u = dr(g[6:9]); w = np.cross(n, u)
                    hu, hw = min(g[9], 1.0), min(g[10], 1.0)
                    cs = [o + su * hu * u + sw * hw * w for su, sw in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
                    ax.add_patch(mp.Polygon([[c[ia], c[ib]] for c in cs], closed=True, fill=True, alpha=0.18 * alpha,
                                            color=col, lw=0))
                    ax.add_patch(mp.Polygon([[c[ia], c[ib]] for c in cs], closed=True, fill=False, alpha=alpha,
                                            color=col, lw=1.1))
                else:
                    c = tf(g[:3])
                    if g[3] > 0:
                        ax.add_patch(mp.Circle((c[ia], c[ib]), g[3], fill=False, color=col, alpha=alpha, lw=1.0))
                    else:
                        ax.plot([c[ia]], [c[ib]], "o", ms=3.2, color=col, alpha=alpha)
                    if alpha == 1.0 and vi == 0:
                        ax.text(c[ia], c[ib], f" {b}.{fn}", fontsize=6, family=T.MONO, color=col, va="bottom")
            if alpha == 1.0:                                      # swing path of points on spin bodies
                for (b, fn), (kind, g) in M.feats.items():
                    if b not in names or fn == "__origin__" or kind != "sphere":
                        continue
                    i = names.index(b)
                    if M.arrays["rmode"][i] != 1:
                        continue
                    from ..compiler.model3d import rot
                    p0 = R[i] @ np.asarray(g[:3]); ax_ = M.arrays["axis"][i]
                    pts = np.array([X[i] + rot(ax_, th) @ p0 for th in np.linspace(0, 2 * math.pi, 97)])
                    ax.plot(pts[:, ia], pts[:, ib], ls=(0, (2, 3)), lw=0.7, color=T.SERIES[i % len(T.SERIES)], alpha=0.6)
            for i, b in enumerate(names):
                col = T.SERIES[i % len(T.SERIES)]
                ax.plot([X[i, ia]], [X[i, ib]], marker="+", ms=7, mew=1.1, color=col, alpha=alpha)
                if alpha == 1.0:
                    ax.text(X[i, ia], X[i, ib], f"  {b}", fontsize=7, weight="bold", color=col, va="top")
                if alpha == 1.0 and M.arrays["rmode"][i] == 1:          # spin axis
                    a_ = M.arrays["axis"][i]; L = 0.6 * max(M.reach.max() if len(M.reach) else 0.05, 0.02)
                    ax.annotate("", xy=(X[i, ia] + L * a_[ia], X[i, ib] + L * a_[ib]), xytext=(X[i, ia], X[i, ib]),
                                arrowprops=dict(arrowstyle="->", color=T.ACCENT, lw=0.9))
        ax.set_aspect("equal", adjustable="datalim"); ax.margins(0.15)
        T.style_ticks(ax); ax.grid(color=T.HAIRLINE, lw=0.5)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.set_title(vt, fontsize=8, color=T.INK, loc="left")
        lab = "xyz"
        ax.set_xlabel(f"{lab[ia]} (m)", fontsize=7, color=T.INK); ax.set_ylabel(f"{lab[ib]} (m)", fontsize=7, color=T.INK)
    notes = [f"{en}: {M.el_info[en]['type']} on {M.el_info[en]['measure']}" for en in M.el_names]
    if any(M.arrays["grav"]):
        g = M.arrays["grav"]; notes.append(f"gravity ({g[0]:.3g}, {g[1]:.3g}, {g[2]:.3g}) m/s^2")
    T.footer(fig, "   ".join(notes)[:420], max_lines=3)
    return T.save(fig, path_no_ext, strict=False)

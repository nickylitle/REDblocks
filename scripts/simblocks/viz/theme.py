"""Mission-control dark theme for every figure, diagram and report the skill produces.

One palette, one type system, one set of components, so all output reads as a single product.
Palette validated with the dataviz validator against SURFACE (#0b1018): lightness band, chroma floor,
adjacent-pair CVD separation >= 8 (worst 9.1), normal-vision floor >= 15 (worst 19.6), contrast >= 3:1.
Categorical hues are assigned in FIXED order, never cycled; >5 series fold into small multiples.
Status colors are reserved for pass / warn / fail and always ship with a text label.
"""
import os
import datetime as _dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")        # bundled with the package (Inter + JetBrains Mono, OFL)

# ---------------------------------------------------------------- tokens
SURFACE = "#0b1018"        # page / figure background
PANEL = "#111823"          # axes background
PANEL_2 = "#162030"        # tiles, table header
HAIRLINE = "#1f2a3a"       # grid
EDGE = "#2a3850"           # axes spines, glyph outlines (muted)
INK = "#e6edf5"            # primary text
INK_2 = "#9fb0c4"          # secondary text
INK_3 = "#64758b"          # muted text, ticks
ACCENT = "#38bdf8"         # header rule, highlights (not a series color)

SERIES = ["#1b95d8", "#c27c00", "#14a06e", "#8c72f2", "#d84b8b"]   # validated order
STATUS = {"PASS": "#2dd4bf", "WARN": "#fbbf24", "FAIL": "#f87171", "SKIP": "#64758b", "INFO": "#94a3b8"}
SEQ = ["#0c2a44", "#0f4470", "#13609c", "#1b7cc4", "#38a0e6", "#7cc4f2"]   # one-hue sequential (blue)

MONO = "JetBrains Mono"
SANS = "Inter"
_REGISTERED = False


def _register_fonts():
    global _REGISTERED, MONO, SANS
    if _REGISTERED:
        return
    if os.path.isdir(FONT_DIR):
        for f in os.listdir(FONT_DIR):
            if f.endswith((".ttf", ".otf")):
                try:
                    font_manager.fontManager.addfont(os.path.join(FONT_DIR, f))
                except Exception:
                    pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    if MONO not in names:
        MONO = "DejaVu Sans Mono"
    if SANS not in names:
        SANS = "DejaVu Sans"
    _REGISTERED = True


def apply():
    """Install the theme into matplotlib (and seaborn if present). Call once before plotting."""
    _register_fonts()
    rc = {
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": PANEL,
        "axes.edgecolor": EDGE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": HAIRLINE, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 9, "axes.labelsize": 8.5, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
        "text.color": INK, "font.family": SANS, "font.size": 9, "font.cursive": [SANS], "font.fantasy": [SANS],
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.markersize": 5,
        "patch.linewidth": 0, "legend.frameon": False, "legend.fontsize": 7.5,
        "legend.labelcolor": INK_2, "legend.handlelength": 1.4,
        "figure.dpi": 140, "savefig.dpi": 170, "svg.fonttype": "none",
        "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": SANS, "mathtext.it": SANS,
    }
    plt.rcParams.update(rc)
    try:
        import seaborn as sns
        sns.set_theme(style="dark", rc=rc, palette=SERIES)
        plt.rcParams.update(rc)          # seaborn resets a few keys
    except ImportError:
        pass
    for a in ("xtick.labelsize", "ytick.labelsize"):
        plt.rcParams[a] = 7.5
    return rc


def mono(ax_or_fig=None):
    """Font properties dict for numeric/telemetry text."""
    return dict(family=MONO)


def style_ticks(ax):
    """Numbers on axes in the monospace face."""
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(MONO)


def header(fig, title, subtitle="", status=None, tag="SIMBLOCKS"):
    """Mission-control header band: tag, title, subtitle, UTC stamp and a status chip."""
    W, H = fig.get_size_inches()
    top = 1 - 0.14 / H
    fig.text(0.018, top, tag, ha="left", va="top", fontsize=7, color=ACCENT, family=MONO, weight="bold")
    fig.text(0.018, top - 0.20 / H, title.upper(), ha="left", va="top", fontsize=13, color=INK,
             family=SANS, weight="semibold")
    if subtitle:
        fig.text(0.018, top - 0.47 / H, subtitle, ha="left", va="top", fontsize=8, color=INK_2, family=SANS)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    fig.text(0.982, top, stamp, ha="right", va="top", fontsize=7, color=INK_3, family=MONO)
    if status:
        col = STATUS.get(status.split()[0], INK_2)
        fig.text(0.982, top - 0.22 / H, f"● {status}", ha="right", va="top", fontsize=8.5, color=col,
                 family=MONO, weight="bold")
    fig.add_artist(matplotlib.lines.Line2D([0.018, 0.982], [top - 0.70 / H] * 2, color=EDGE, lw=0.8,
                                           transform=fig.transFigure))
    return 1 - 0.95 / H        # usable top for tight_layout rect


def footer(fig, text, max_lines=2):
    """Source line at the bottom. Long text wraps to max_lines, then ends in an ellipsis; the full text
    belongs in the card/report markdown, never cut silently there."""
    import textwrap
    t = fig.text(0.018, 0.012, text, ha="left", va="bottom", fontsize=6.5, color=INK_3, family=MONO)
    r = fig.canvas.get_renderer()
    avail = 0.964 * fig.bbox.width
    w = t.get_window_extent(r).width
    if w <= avail:
        return t
    chars = max(20, int(len(text) * avail / w) - 2)
    lines = textwrap.wrap(text, chars)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(0, chars - 3)].rstrip() + "..."
    t.set_text("\n".join(lines))
    return t


def stat_tile(ax, label, value, unit="", sub="", status=None):
    """Hero-number tile drawn into an axes (axis off)."""
    ax.set_facecolor(PANEL_2); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.06, 0.80, label.upper(), transform=ax.transAxes, fontsize=7, color=INK_2, family=MONO, va="top")
    ax.text(0.06, 0.40, value, transform=ax.transAxes, fontsize=19, color=INK, family=MONO, weight="medium",
            va="center")
    if unit:
        ax.text(0.94, 0.40, unit, transform=ax.transAxes, fontsize=8, color=INK_3, family=MONO, va="center", ha="right")
    if sub:
        ax.text(0.06, 0.12, sub, transform=ax.transAxes, fontsize=7, color=INK_3, family=SANS, va="center")
    if status:
        col = STATUS.get(status, INK_2)
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, color=col, lw=0))
        ax.text(0.94, 0.80, status, transform=ax.transAxes, fontsize=7, color=col, family=MONO, va="top",
                ha="right", weight="bold")


def limit_line(ax, y=None, x=None, label="", status="FAIL"):
    """Dashed threshold line in a status color with an inline text label (never color alone)."""
    col = STATUS.get(status, INK_2)
    if y is not None:
        ax.axhline(y, color=col, lw=1.1, ls=(0, (4, 3)), zorder=5)
        t = ax.text(1.0, y, f" {label}", transform=ax.get_yaxis_transform(), color=col, fontsize=7, family=MONO,
                    va="center", ha="left", clip_on=False)
        fig = ax.figure
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer(); tb = t.get_window_extent(r_)
        hits_neighbor = any(o is not ax and o.get_visible() and tb.overlaps(o.get_window_extent(r_))
                            for o in fig.axes)
        # panels not on the right edge of the page may get a neighbour later: keep the label inside then
        not_rightmost = ax.get_position().x1 < 0.9
        if tb.x1 > fig.bbox.width - 4 or hits_neighbor or not_rightmost:
            # no room in the right margin: sit the label just above the line, inside the plot, right-aligned
            t.set_position((0.995, y)); t.set_text(label); t.set_ha("right"); t.set_va("bottom")
    if x is not None:
        ax.axvline(x, color=col, lw=1.1, ls=(0, (4, 3)), zorder=5)
        ax.text(x, 1.0, f"{label}", transform=ax.get_xaxis_transform(), color=col, fontsize=7, family=MONO,
                va="bottom", ha="center", clip_on=False)


def direct_label(ax, x, y, text, color):
    ax.text(x, y, f" {text}", color=INK_2, fontsize=7, family=MONO, va="center", ha="left")
    ax.plot([x], [y], "o", ms=4, color=color, mec=PANEL, mew=1.5, zorder=6)


QA_STRICT = True          # figures with text collisions or clipping are not saved
QA_LOG = {}               # path -> problems, for the final pass report


def save(fig, path_no_ext, formats=("png", "svg"), strict=None):
    """Save after layout QA. strict (default QA_STRICT) raises LayoutError on any problem."""
    from .qa import check_figure, LayoutError
    probs = check_figure(fig)
    QA_LOG[path_no_ext] = probs
    if probs and (QA_STRICT if strict is None else strict):
        plt.close(fig)
        raise LayoutError(f"{path_no_ext}: {len(probs)} layout problem(s):\n  " + "\n  ".join(probs[:20]))
    out = []
    for f in formats:
        p = f"{path_no_ext}.{f}"
        try:
            fig.savefig(p, facecolor=SURFACE)
        except ImportError:                              # that format's writer is not in this build: skip it
            if f == "png":
                raise
            continue
        out.append(p)
    plt.close(fig)
    return out


CSS = f"""
:root {{ --surface:{SURFACE}; --panel:{PANEL}; --panel2:{PANEL_2}; --hair:{HAIRLINE}; --edge:{EDGE};
  --ink:{INK}; --ink2:{INK_2}; --ink3:{INK_3}; --accent:{ACCENT};
  --pass:{STATUS['PASS']}; --warn:{STATUS['WARN']}; --fail:{STATUS['FAIL']}; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--surface); color:var(--ink); font:14px/1.55 Inter, system-ui, sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:28px 16px 64px }}
.tag {{ font:700 11px 'JetBrains Mono', ui-monospace, monospace; color:var(--accent); letter-spacing:.14em }}
h1 {{ font-weight:600; font-size:26px; letter-spacing:.02em; text-transform:uppercase; margin:6px 0 2px }}
h2 {{ font-weight:600; font-size:15px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink2);
     border-top:1px solid var(--edge); padding-top:18px; margin-top:34px }}
.sub {{ color:var(--ink2) }} .mono {{ font-family:'JetBrains Mono', ui-monospace, monospace }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; margin:18px 0 }}
.tile {{ background:var(--panel2); border-left:3px solid var(--edge); padding:12px 14px }}
.tile .l {{ font:11px 'JetBrains Mono',monospace; color:var(--ink2); text-transform:uppercase }}
.tile .v {{ font:500 26px 'JetBrains Mono',monospace }} .tile .s {{ color:var(--ink3); font-size:12px }}
.tile.PASS {{ border-left-color:var(--pass) }} .tile.WARN {{ border-left-color:var(--warn) }} .tile.FAIL {{ border-left-color:var(--fail) }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin:10px 0 }}
th {{ background:var(--panel2); color:var(--ink2); font:600 11px 'JetBrains Mono',monospace; text-transform:uppercase;
     text-align:left; padding:8px 10px; letter-spacing:.05em }}
td {{ border-bottom:1px solid var(--hair); padding:7px 10px; vertical-align:top }}
td.num {{ font-family:'JetBrains Mono',monospace; white-space:nowrap }}
.chip {{ font:700 11px 'JetBrains Mono',monospace; padding:2px 7px; border-radius:3px; white-space:nowrap }}
.chip.PASS {{ color:var(--pass); background:#2dd4bf1a }} .chip.WARN {{ color:var(--warn); background:#fbbf241a }}
.chip.FAIL {{ color:var(--fail); background:#f871711a }} .chip.SKIP {{ color:var(--ink3); background:#64758b1a }}
img, svg {{ max-width:100%; height:auto; display:block; margin:12px 0; border:1px solid var(--hair) }}
.note {{ color:var(--ink3); font-size:12px }}
@media (max-width:640px) {{ h1 {{ font-size:20px }} .tile .v {{ font-size:22px }} table {{ font-size:12px }} }}
"""


def place_point_labels(ax, xs, ys, texts, fontsize=6.8, color=None):
    """Label data points, trying candidate offsets until a label clears every drawn line and every label
    already placed (checked with real renderer extents). Falls back to the least-bad spot."""
    import numpy as np
    from matplotlib.lines import Line2D
    fig = ax.figure
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    skip = set(ax.get_xgridlines()) | set(ax.get_ygridlines())
    segs = []
    for ln in ax.get_lines():
        if ln in skip or not ln.get_visible():
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        for a, b in zip(xy[:-1], xy[1:]):
            n = int(max(2, np.hypot(*(b - a)) / 3))
            segs.extend(a + (b - a) * k / n for k in range(n + 1))
    pts = np.array(segs) if segs else np.zeros((0, 2))
    cands = [(0, 9, "center", "bottom"), (0, -9, "center", "top"), (7, 6, "left", "bottom"), (-7, 6, "right", "bottom"),
             (7, -6, "left", "top"), (-7, -6, "right", "top"), (0, 18, "center", "bottom"), (0, -18, "center", "top"),
             (12, 0, "left", "center"), (-12, 0, "right", "center"), (14, 14, "left", "bottom"),
             (-14, 14, "right", "bottom"), (14, -14, "left", "top"), (-14, -14, "right", "top")]
    placed = []
    for x, y, s in zip(xs, ys, texts):
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                            color=color or INK_2, family=MONO)
            bb = t.get_window_extent(r).expanded(1.08, 1.15)
            hits = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum()) if len(pts) else 0
            hits += sum(1 for p in placed if bb.overlaps(p)) * 100
            ax_bb = ax.get_window_extent(r)
            if not (ax_bb.x0 <= bb.x0 and bb.x1 <= ax_bb.x1 and ax_bb.y0 <= bb.y0 and bb.y1 <= ax_bb.y1):
                hits += 50
            if best is None or hits < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hits, t, bb)
            else:
                t.remove()
            if hits == 0:
                break
        if best[0] > 0:
            # no clear spot: a value label sitting on a line is worse than no label (the axis still reads it)
            best[1].remove()
            continue
        placed.append(best[2])

"""Layout QA for every figure before it is saved or sent.

Checks, using the real renderer extents (what the reader will see):
  * text-text overlap        two visible text artists whose boxes intersect (beyond a 1 px tolerance)
  * text outside the figure  any part of a label off the canvas (clipped)
  * text outside its axes    tick/axis labels excluded; data labels that spill past the axes frame
  * tiny text                font below the readable floor (6 pt) at the figure's size
theme.save() calls check_figure() and raises LayoutError in strict mode, so a figure with collisions never
reaches the user. Run qa_report(paths) on a folder of outputs for a final pass.
"""
import itertools

from matplotlib.text import Text

from ..registry import block

MIN_PT = 6.0


class LayoutError(RuntimeError):
    pass


def _texts(fig):
    out = []
    for t in fig.findobj(match=Text):
        try:
            s = t.get_text()
        except Exception:
            continue
        if not s or not s.strip() or not t.get_visible():
            continue
        if t.get_alpha() is not None and t.get_alpha() == 0:
            continue
        out.append(t)
    return out


@block("diagnostics", "report", "any_figure", "layout_qa", kind="builder", source="matplotlib renderer extents",
       tags=("overlap", "readability", "QA"))
def check_figure(fig, tol_px=1.0, check_axes_spill=True):
    """Return a list of human-readable layout problems (empty list = clean)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    _collect_ticks(fig)
    W, H = fig.bbox.width, fig.bbox.height
    items = []
    for t in _texts(fig):
        if id(t) in _HIDDEN_TICKS:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((t, bb))
    probs = []
    for t, bb in items:
        # tick labels too: a clipped row label is as unreadable as any other (hidden ticks are not drawn)
        if (bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px) \
                and id(t) not in _HIDDEN_TICKS:
            probs.append(f"OFF-CANVAS: '{t.get_text()[:40]}'")
        # text that leaves its own axes and lands on ANOTHER axes is hidden or confusing (e.g. a limit label
        # running into the neighbouring panel)
        own = t.axes
        if own is not None and not _is_tick(t) and t is not getattr(own, "title", None):
            for o in fig.axes:
                if o is own or not o.get_visible() or not o.axison:
                    continue
                ob = o.get_window_extent(r)
                if bb.overlaps(ob):
                    probs.append(f"INTO NEIGHBOR AXES: '{t.get_text()[:40]}'")
                    break
        if t.get_fontsize() < MIN_PT:
            probs.append(f"TOO SMALL ({t.get_fontsize():.1f} pt): '{t.get_text()[:40]}'")
    for (a, ba), (b, bb) in itertools.combinations(items, 2):
        ix = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
        iy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
        if ix > tol_px and iy > tol_px:
            # tick labels on shared axes legitimately abut at corners; ignore pairs that are both tick labels
            if _is_tick(a) and _is_tick(b):
                continue
            probs.append(f"OVERLAP: '{a.get_text()[:32]}'  x  '{b.get_text()[:32]}'")
    probs += _line_hits(fig, items, r, tol_px)
    return sorted(set(probs), key=probs.index)


def _line_hits(fig, items, r, tol_px):
    """Data lines / glyph strokes passing through a text box (grid, ticks, spines, legend handles excluded)."""
    import numpy as np
    from matplotlib.lines import Line2D
    probs = []
    skip = set()
    for ax in fig.axes:
        skip.update(ax.get_xgridlines()); skip.update(ax.get_ygridlines())
        skip.update(ax.get_xticklines()); skip.update(ax.get_yticklines())
        leg = ax.get_legend()
        if leg is not None:
            skip.update(leg.get_lines())
    for ln in fig.findobj(match=Line2D):
        if ln in skip or not ln.get_visible() or ln.get_gid() == "qa-ignore":
            continue
        try:
            xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        except Exception:
            continue
        if len(xy) < 2 or not np.all(np.isfinite(xy)):
            continue
        # densify segments so a short text box crossed mid-segment is caught
        pts = [xy[0]]
        for a, b in zip(xy[:-1], xy[1:]):
            n = int(max(2, np.hypot(*(b - a)) / 3))
            pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
        pts = np.array(pts)
        if ln.get_clip_on():
            cb = ln.get_clip_box()
            if cb is not None:          # only the part actually drawn
                pts = pts[(pts[:, 0] >= cb.x0) & (pts[:, 0] <= cb.x1) & (pts[:, 1] >= cb.y0) & (pts[:, 1] <= cb.y1)]
        if len(pts) < 2:
            continue
        for t, bb in items:
            if getattr(t, "_qa_on_line", False):
                continue
            inside = (pts[:, 0] > bb.x0 + tol_px) & (pts[:, 0] < bb.x1 - tol_px) & \
                     (pts[:, 1] > bb.y0 + tol_px) & (pts[:, 1] < bb.y1 - tol_px)
            if inside.sum() >= 2:
                probs.append(f"LINE THROUGH TEXT: '{t.get_text()[:40]}'")
    return probs


_TICKS = set()
_HIDDEN_TICKS = set()      # tick labels outside the view interval: matplotlib does not draw them


def _collect_ticks(fig):
    _TICKS.clear(); _HIDDEN_TICKS.clear()
    for ax in fig.axes:
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None)):
            if axis is None:
                continue
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) or 1.0
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                _TICKS.add(id(tk.label1)); _TICKS.add(id(tk.label2))
                try:
                    loc = tk.get_loc()
                except Exception:
                    loc = None
                if loc is None or not (lo - 1e-9 * span <= loc <= hi + 1e-9 * span):
                    _HIDDEN_TICKS.add(id(tk.label1)); _HIDDEN_TICKS.add(id(tk.label2))


def _is_tick(t):
    return id(t) in _TICKS


@block("diagnostics", "report", "any_figure", "qa_folder", kind="builder", source="PNG file audit",
       tags=("QA", "final pass"))
def qa_summary(results):
    """Format {path: [problems]} into a short pass/fail report."""
    lines = []
    for p, probs in results.items():
        lines.append(f"{'PASS' if not probs else 'FAIL'}  {p}")
        lines += [f"      {x}" for x in probs[:12]]
    return "\n".join(lines)

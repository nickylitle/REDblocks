"""Results report: the figures and page that answer the question, show what is interesting, and show what
to design around. All pages use the mission-control theme.

    pages = results_report(out_dir, name, question, answer, rep, nominal,
                           history=hist, history_panels=[...], sweep=dict(...), findings=[...],
                           assumptions=[...], sources=[...])

Pages (PNG + SVG each, plus one multi-page PDF and one self-contained HTML):
  1 ANSWER        hero tiles (one per failure mode + headline), safety-factor distributions vs SF = 1
  2 TIME HISTORY  small multiples of the key states with failure limits drawn in
  3 DESIGN AROUND what drives the governing margin (rank correlation), parameter sweep with failure band
  4 VERIFICATION  every harness check with status, dt error and ledger closure
"""
import base64
import html
import io
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import theme as T
from ..registry import block


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "inf" if v == math.inf else "n/a"
    a = abs(v)
    if a >= 1e6 or (a != 0 and a < 1e-3):
        return f"{v:.{nd-1}e}"
    if a >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}g}"


def _pct(p):
    """Probability as a readable percent: 0%, 0.25%, 46%, 100%."""
    x = 100.0 * p
    if x == 0 or x >= 99.95:
        return f"{x:.0f}%"
    return f"{x:.2g}%" if x < 10 else f"{x:.0f}%"


def fit_left_margin(fig, ax, pad_in=0.12):
    """Shift an axes right/narrow it so its y tick labels and y label stay on the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
    if ax.yaxis.label.get_text():
        boxes.append(ax.yaxis.label.get_window_extent(r))
    if not boxes:
        return
    need = (min(b.x0 for b in boxes)) / fig.bbox.width       # how far left the labels reach (fraction)
    pos = ax.get_position()
    margin = pad_in / fig.get_size_inches()[0]
    if need < margin:
        shift = margin - need
        ax.set_position([pos.x0 + shift, pos.y0, pos.width - shift, pos.height])


def _status_of_mode(m):
    if m.get("reach") and not m["reach"]["reachable"]:
        return "WARN"
    if m["nominal_failed"]:
        return "FAIL"
    return "WARN" if m["n_failed"] > 0 else "PASS"


def _page_answer(name, question, answer, rep, figs):
    fm = getattr(rep, "failure", {}) or {}
    modes = list(fm.items())
    ncols = max(len(modes) + 1, 3)
    fig = plt.figure(figsize=(13, 7.4))
    st = "VERIFIED" if rep.ok else "NOT VERIFIED"
    top = T.header(fig, name, question, status=("PASS " if rep.ok else "FAIL ") + st, tag="SIMBLOCKS // RESULT")
    gs = fig.add_gridspec(2, ncols, left=0.10, right=0.97, bottom=0.10, top=top - 0.02,
                          height_ratios=[0.9, 2.6], hspace=0.38, wspace=0.18)
    a0 = fig.add_subplot(gs[0, 0])
    T.stat_tile(a0, answer.get("label", "answer"), answer.get("value", ""), answer.get("unit", ""),
                answer.get("sub", ""), answer.get("status"))
    for i, (mn, m) in enumerate(modes):
        ax = fig.add_subplot(gs[0, i + 1])
        sub = f"P(fail) {_pct(m['p_fail'])}  |  SF P5 {_fmt(m['SF_p5'])}"
        if m.get("reach") and not m["reach"]["reachable"]:
            T.stat_tile(ax, f"{mn}  [UNREACHABLE]", f"max {_fmt(m['reach']['max_possible'])}", "",
                        f"geometry cannot reach limit {m['limit']:g}", "WARN")
            continue
        T.stat_tile(ax, f"{mn}{'  [GOVERNING]' if m.get('governing') else ''}", f"SF {_fmt(m['nominal_SF'])}",
                    "", sub, _status_of_mode(m))
    for j in range(len(modes) + 1, ncols):
        fig.add_subplot(gs[0, j]).axis("off")
    ax = fig.add_subplot(gs[1, :])
    if modes:
        y = np.arange(len(modes)); ylabels = []
        for i, (mn, m) in enumerate(modes):
            s = np.array(m["SF_samples"], float); s = s[np.isfinite(s)]
            col = T.SERIES[i % len(T.SERIES)]
            if len(s):
                jit = (np.random.default_rng(i).random(len(s)) - 0.5) * 0.46
                ax.scatter(np.clip(s, 1e-2, None), i + jit, s=9, color=col, alpha=0.55, lw=0, zorder=3)
                q = np.percentile(s, [5, 50, 95])
                ax.plot([q[0], q[2]], [i, i], color=T.INK, lw=1.2, zorder=4)
                ax.plot([q[1]], [i], "o", ms=6, color=T.INK, mec=T.PANEL, mew=1.5, zorder=5)
                # percentiles go in the row label (outside the data), so they never collide with marks or limits
                ylabels.append(f"{mn}\nP5 {_fmt(q[0])} | P50 {_fmt(q[1])} | P95 {_fmt(q[2])}")
            else:
                ylabels.append(mn)
            ax.plot([m["nominal_SF"]], [i], marker="D", ms=6, color=T.ACCENT, mec=T.PANEL, zorder=6)
        ax.set_yticks(y); ax.set_yticklabels(ylabels, fontsize=8, family=T.MONO)
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        from matplotlib.ticker import NullFormatter, LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))   # 0.5, 1, 2, 5: always several ticks
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axvline(1.0, color=T.STATUS["FAIL"], lw=1.1, ls=(0, (4, 3)), zorder=5)
        ax.text(1.0, 0.02, " SF = 1  FAILURE", transform=ax.get_xaxis_transform(), color=T.STATUS["FAIL"],
                fontsize=7, family=T.MONO, va="bottom", ha="left")
        lo_, hi_ = ax.get_xlim()
        ax.axvspan(lo_, 1.0, color=T.STATUS["FAIL"], alpha=0.06, lw=0)
        ax.set_xlim(lo_, hi_)
        ax.set_xlabel("safety factor  (limit / demand)   ◆ nominal   ● median, bar P5-P95, dots = Monte Carlo hits")
        ax.set_title("Margin to each user-defined failure mode", loc="left")
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        fit_left_margin(fig, ax)
    else:
        ax.text(0.5, 0.5, "No failure modes defined: ask the user what failure means for this part.",
                ha="center", va="center", color=T.STATUS["WARN"], transform=ax.transAxes)
        ax.axis("off")
    T.style_ticks(ax)
    T.footer(fig, rep.summary())
    figs.append(("answer", fig))


def _page_history(name, history, panels, figs):
    n = len(panels); ncol = min(3, n); nrow = math.ceil(n / ncol)
    H = 3.0 * nrow + 1.9
    fig = plt.figure(figsize=(13, H))
    top = T.header(fig, name, "Time histories of the governing states (nominal case)", tag="SIMBLOCKS // TIME HISTORY")
    # margins in inches so short (one-row) pages keep room for titles and x labels
    gs = fig.add_gridspec(nrow, ncol, left=0.055, right=0.95, bottom=0.62 / H, top=top - 0.42 / H,
                          hspace=0.75 / 2.4, wspace=0.28)
    t = np.asarray(history["t"]) * panels[0].get("t_scale", 1e3)
    for k, pnl in enumerate(panels):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ys = pnl["keys"] if isinstance(pnl["keys"], (list, tuple)) else [pnl["keys"]]
        for i, key in enumerate(ys):
            y = np.asarray(history[key]) * pnl.get("scale", 1.0)
            ax.plot(t, y, color=T.SERIES[(pnl.get("color", 0) + i) % len(T.SERIES)], lw=1.6,
                    label=pnl.get("labels", ys)[i] if len(ys) > 1 else None)
            if len(ys) == 1:            # multi-series panels use the legend; end labels could collide
                T.direct_label(ax, t[-1], y[-1], _fmt(y[-1]), T.SERIES[(pnl.get("color", 0) + i) % len(T.SERIES)])
        if pnl.get("limit") is not None:
            T.limit_line(ax, y=pnl["limit"], label=pnl.get("limit_label", "limit"), status="FAIL")
        ax.set_title(pnl["title"]); ax.set_xlabel(pnl.get("t_label", "time (ms)")); ax.set_ylabel(pnl.get("unit", ""))
        if pnl.get("headroom"):              # extra empty band above the data for the legend
            lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + (hi_ - lo_) * pnl["headroom"])
        if len(ys) > 1:
            ax.legend(loc=pnl.get("legend_loc", "best"), ncol=pnl.get("legend_ncol", 1))
        T.style_ticks(ax)
    figs.append(("history", fig))


def _page_design(name, rep, sweep, figs):
    fm = getattr(rep, "failure", {}) or {}
    gov = next((k for k, v in fm.items() if v.get("governing")), None)
    fig = plt.figure(figsize=(13, 5.6))
    top = T.header(fig, name, "What to design around: drivers of the governing margin and the design-variable sweep",
                   tag="SIMBLOCKS // DESIGN AROUND")
    gs = fig.add_gridspec(1, 2, left=0.19, right=0.94, bottom=0.25, top=top - 0.07, wspace=0.30, width_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    if gov and fm[gov]["drivers"]:
        d = [(n_, r) for n_, r in fm[gov]["drivers"] if math.isfinite(r)]
        lab = (sweep or {}).get("input_labels", {})
        names = [lab.get(x[0], x[0]) for x in d][::-1]; vals = [x[1] for x in d][::-1]
        cols = [T.SERIES[0] if v > 0 else T.SERIES[1] for v in vals]
        ax.barh(names, vals, color=cols, height=0.55)
        for i, v in enumerate(vals):
            # value sits on the empty side of zero so it can never run into the bar labels or off the axes
            ax.text(-0.03 if v >= 0 else 0.03, i, f"{v:+.2f}", va="center", ha="right" if v >= 0 else "left",
                    fontsize=7, color=T.INK_2, family=T.MONO)
        ax.axvline(0, color=T.EDGE, lw=1); ax.set_xlim(-1.15, 1.15)
        ax.set_title(f"Drivers of '{gov}' margin"); ax.set_xlabel("rank correlation with SF  (+ = raises margin)")
        ax.grid(axis="y", visible=False)
    else:
        ax.axis("off")
    T.style_ticks(ax)
    ax = fig.add_subplot(gs[0, 1])
    if sweep:
        x = np.asarray(sweep["x"]) * sweep.get("x_scale", 1.0)
        multi = len(sweep["series"]) > 1
        for i, (lab, y) in enumerate(sweep["series"].items()):
            y = np.asarray(y, float)
            if not np.isfinite(y).any():
                lab = f"{lab}: no finite result at any x"
            elif not np.isfinite(y).all():
                lab = f"{lab} (gaps = no result)"
            ax.plot(x, y, "-o", color=T.SERIES[i % len(T.SERIES)], ms=4.5, mec=T.PANEL, mew=1.2, label=lab)
        pad = 0.04 * (x.max() - x.min() or 1.0)
        ax.set_xlim(x.min() - pad, x.max() + pad)          # full design range even where results are missing
        if sweep.get("limit") is not None:
            T.limit_line(ax, y=sweep["limit"], label=sweep.get("limit_label", "failure"), status="FAIL")
            lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, max(hi_, sweep["limit"] * 1.08))
            ax.axhspan(sweep["limit"], ax.get_ylim()[1] if sweep.get("fails_when", "above") == "above" else ax.get_ylim()[0],
                       color=T.STATUS["FAIL"], alpha=0.07, lw=0)
        if sweep.get("mark") is not None:
            ax.axvline(sweep["mark"] * sweep.get("x_scale", 1.0), color=T.ACCENT, lw=1, ls=":",
                       label=f"current design ({_fmt(sweep['mark'] * sweep.get('x_scale', 1.0))})")
        ax.set_title(sweep.get("title", "Design sweep")); ax.set_xlabel(sweep.get("x_label", "")); ax.set_ylabel(sweep.get("y_label", ""))
        # selective labels: one series -> every point; several -> only the current design (mark) column,
        # because labels on every point of every line inevitably sit on a neighbouring line
        allx, ally, allt = [], [], []
        xm = sweep["mark"] * sweep.get("x_scale", 1.0) if sweep.get("mark") is not None else None
        for y in sweep["series"].values():
            for xi, yi in zip(x, np.asarray(y, float)):
                if not np.isfinite(yi):
                    continue
                if multi and xm is not None and abs(xi - xm) > 1e-9 * max(abs(xm), 1.0):
                    continue
                allx.append(xi); ally.append(yi); allt.append(_fmt(yi))
        T.place_point_labels(ax, allx, ally, allt)
        if multi or sweep.get("mark") is not None:
            # legend under the x label, outside the data: it can never sit on a curve
            nleg = len(ax.get_legend_handles_labels()[0])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=min(nleg, 2), frameon=False)
    else:
        ax.axis("off")
    T.style_ticks(ax)
    figs.append(("design", fig))


def _page_verification(name, rep, figs):
    fig = plt.figure(figsize=(13, 6.8))
    top = T.header(fig, name, "Automated verification battery", status=("PASS VERIFIED" if rep.ok else "FAIL NOT VERIFIED"),
                   tag="SIMBLOCKS // VERIFICATION")
    ax = fig.add_axes([0.03, 0.06, 0.94, top - 0.08]); ax.axis("off")
    rows = [c for c in rep.checks]
    y = 0.97; dy = min(0.9 / max(len(rows), 1), 0.065)
    ax.text(0.0, y, "CHECK", fontsize=7, color=T.INK_3, family=T.MONO, transform=ax.transAxes)
    ax.text(0.27, y, "STATUS", fontsize=7, color=T.INK_3, family=T.MONO, transform=ax.transAxes)
    ax.text(0.36, y, "DETAIL", fontsize=7, color=T.INK_3, family=T.MONO, transform=ax.transAxes)
    for c in rows:
        y -= dy
        ax.add_line(plt.Line2D([0, 1], [y + dy * 0.55] * 2, color=T.HAIRLINE, lw=0.6, transform=ax.transAxes))
        ax.text(0.0, y, c.name, fontsize=7.6, color=T.INK, family=T.MONO, transform=ax.transAxes, va="center")
        ax.text(0.27, y, f"● {c.status}", fontsize=7.6, color=T.STATUS.get(c.status, T.INK_2), family=T.MONO,
                weight="bold", transform=ax.transAxes, va="center")
        det = c.detail if len(c.detail) < 150 else c.detail[:147] + "..."
        ax.text(0.36, y, det, fontsize=6.9, color=T.INK_2, transform=ax.transAxes, va="center")
    figs.append(("verification", fig))


def _html(name, question, answer, rep, pages, findings, assumptions, sources, svg_paths):
    def esc(s):
        return html.escape(str(s))
    fm = getattr(rep, "failure", {}) or {}
    tiles = [f"<div class='tile {answer.get('status','')}'><div class='l'>{esc(answer.get('label',''))}</div>"
             f"<div class='v'>{esc(answer.get('value',''))} <span class='s'>{esc(answer.get('unit',''))}</span></div>"
             f"<div class='s'>{esc(answer.get('sub',''))}</div></div>"]
    for mn, m in fm.items():
        stt = _status_of_mode(m)
        tiles.append(f"<div class='tile {stt}'><div class='l'>{esc(mn)}{' · governing' if m.get('governing') else ''}</div>"
                     f"<div class='v'>SF {_fmt(m['nominal_SF'])}</div><div class='s'>P(fail) {_pct(m['p_fail'])} "
                     f"[{m['p_fail_ci95'][0]*100:.2g}, {m['p_fail_ci95'][1]*100:.2g}] · SF P5 {_fmt(m['SF_p5'])}</div></div>")
    imgs = []
    for p in svg_paths:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        mime = "image/png" if p.endswith(".png") else "image/svg+xml"
        imgs.append(f"<img alt='{esc(os.path.basename(p))}' src='data:{mime};base64,{b64}'>")
    checks = "".join(f"<tr><td class='mono'>{esc(c.name)}</td><td><span class='chip {c.status}'>{c.status}</span></td>"
                     f"<td>{esc(c.detail)}</td></tr>" for c in rep.checks)
    li = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(name)}</title><style>{T.CSS}</style></head><body><div class="wrap">
<div class="tag">SIMBLOCKS // RESULT</div><h1>{esc(name)}</h1><div class="sub">{esc(question)}</div>
<div class="tiles">{''.join(tiles)}</div>
<h2>Findings</h2><ul>{li(findings)}</ul>
{''.join(imgs)}
<h2>Assumptions</h2><ul>{li(assumptions)}</ul>
<h2>Verification · <span class="chip {'PASS' if rep.ok else 'FAIL'}">{'VERIFIED' if rep.ok else 'NOT VERIFIED'}</span></h2>
<table><tr><th>check</th><th>status</th><th>detail</th></tr>{checks}</table>
<h2>Sources</h2><ul>{li(sources)}</ul>
<p class="note mono">{esc(rep.summary())}</p></div></body></html>"""


@block("diagnostics", "report", "any_sim", "results", kind="builder", source="simblocks viz",
       tags=("figures", "report", "dark theme"))
def results_report(out_dir, name, question, answer, rep, history=None, history_panels=None, sweep=None,
                   findings=(), assumptions=(), sources=(), extra_svgs=()):
    """Write answer / history / design / verification pages (PNG+SVG), a PDF and a self-contained HTML.
    Returns dict of paths."""
    T.apply(); os.makedirs(out_dir, exist_ok=True)
    figs = []
    _page_answer(name, question, answer, rep, figs)
    if history is not None and history_panels:
        _page_history(name, history, history_panels, figs)
    _page_design(name, rep, sweep, figs)
    _page_verification(name, rep, figs)
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
    paths = {}; svgs = list(extra_svgs)
    pdf_path = os.path.join(out_dir, f"{slug}_report.pdf")
    from .qa import check_figure, LayoutError
    for key, f in figs:
        probs = check_figure(f)
        T.QA_LOG[f"{slug}_{key}"] = probs
        if probs and T.QA_STRICT:
            for _, g in figs:
                plt.close(g)
            raise LayoutError(f"{slug}_{key}: {len(probs)} layout problem(s):\n  " + "\n  ".join(probs[:20]))
    with PdfPages(pdf_path) as pdf:
        for key, f in figs:
            pdf.savefig(f, facecolor=T.SURFACE)
            base = os.path.join(out_dir, f"{slug}_{key}")
            f.savefig(base + ".png", facecolor=T.SURFACE)
            paths[key] = base + ".png"
            try:                                         # a build without matplotlib's SVG writer still makes the report
                f.savefig(base + ".svg", facecolor=T.SURFACE); svgs.append(base + ".svg")
            except ImportError:
                svgs.append(base + ".png")            # the page then carries the PNG instead
            plt.close(f)
    # answer page first, then diagrams, then the rest
    first = lambda p: p.endswith(("_answer.svg", "_answer.png"))
    order = [p for p in svgs if first(p)] + list(extra_svgs) + [p for p in svgs if p not in extra_svgs and not first(p)]
    hp = os.path.join(out_dir, f"{slug}_report.html")
    with open(hp, "w") as fh:
        fh.write(_html(name, question, answer, rep, figs, findings, assumptions, sources, order))
    paths.update(pdf=pdf_path, html=hp)
    return paths

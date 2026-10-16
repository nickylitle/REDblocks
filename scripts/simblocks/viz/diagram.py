"""Simulation diagrams composed from the glyph library.

Schematic  : physical lumped-parameter network (Simscape-style). Bodies are nodes; springs, sliders,
             contacts, dashpots are two-terminal elements between them. Every element is labelled with
             its name, its computed value and the block that produced it, so the model can be checked by eye.
SignalFlow : per-time-step computation graph (Simulink-style): state -> geometry -> constitutive -> forces ->
             integrator -> state, with the blocks used in each stage.

Both render to PNG + SVG in the mission-control theme.
"""
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

from . import theme as T
from . import glyphs as G

from ..registry import block


class Schematic:
    """Physical-network diagram. Positions are in grid units (1 = one cell)."""

    def __init__(self, title, subtitle=""):
        self.title, self.subtitle = title, subtitle
        self.bodies = {}; self.links = []; self.probes = []; self.notes = []

    def body(self, key, kind="mass", label=None, info=(), pos=None, **kw):
        self.bodies[key] = dict(kind=kind, label=key if label is None else label, info=list(info), pos=pos, kw=kw)
        return self

    def link(self, a, b, kind="spring", label="", value="", source="", lane=0.0):
        self.links.append(dict(a=a, b=b, kind=kind, label=label, value=value, source=source, lane=lane))
        return self

    def probe(self, at, label):
        self.probes.append((at, label)); return self

    def note(self, text):
        self.notes.append(text); return self

    # -------------------------------------------------------------- layout
    def _chain_order(self):
        """Order non-ground bodies so linked bodies sit next to each other (walk the link graph from an end)."""
        keys = [k for k, b in self.bodies.items() if b["kind"] != "ground" and b["pos"] is None]
        adj = {k: [] for k in keys}
        for l in self.links:
            if l["a"] in adj and l["b"] in adj:
                adj[l["a"]].append(l["b"]); adj[l["b"]].append(l["a"])
        order, seen = [], set()
        while len(order) < len(keys):
            rest = [k for k in keys if k not in seen]
            start = min(rest, key=lambda k: (len([n for n in adj[k] if n not in seen]), keys.index(k)))
            stack = [start]
            while stack:
                k = stack.pop()
                if k in seen:
                    continue
                seen.add(k); order.append(k)
                nxt = sorted([n for n in adj[k] if n not in seen], key=keys.index, reverse=True)
                stack.extend(nxt)
        return order

    def _layout(self):
        x = 0.0
        order = self._chain_order()
        for k in order:
            b = self.bodies[k]
            b["pos"] = (x, 0.0); x += 3.0 if b["kind"] == "point" else 5.4
        # a link that skips over a body would be drawn through it: route it on its own lane above the row
        idx = {k: i for i, k in enumerate(order)}
        lane_n = 0
        for l in self.links:
            if l["a"] in idx and l["b"] in idx and abs(idx[l["a"]] - idx[l["b"]]) > 1 and not l["lane"]:
                lane_n += 1
                l["lane"] = 1.5 * lane_n * (1 if idx[l["b"]] > idx[l["a"]] else -1)
        # several elements between the same two bodies (e.g. drag + friction to ground) get side-by-side lanes
        groups = {}
        for l in self.links:
            groups.setdefault(frozenset((l["a"], l["b"])), []).append(l)
        for grp in groups.values():
            if len(grp) > 1:
                free = [l for l in grp if not l["lane"]]
                for i, l in enumerate(free):
                    l["lane"] = 1.4 * (i - (len(free) - 1) / 2.0)
        for k, b in self.bodies.items():
            if b["pos"] is None:            # ground: under the first body it connects to
                other = next((l["a"] if l["b"] == k else l["b"]) for l in self.links if k in (l["a"], l["b"]))
                ox, oy = self.bodies[other]["pos"]
                b["pos"] = (ox, oy - 2.4)

    def _anchor(self, key, toward):
        b = self.bodies[key]; c = np.array(b["pos"], float); d = np.array(toward, float) - c
        half = {"mass": (_box_w(b["label"]) / 2, 0.31), "rotor": (0.45, 0.45), "wheel": (0.36, 0.36), "motor": (0.32, 0.32),
                "face": (0.15, 0.15), "ground": (0.0, 0.0), "point": (0.0, 0.0)}.get(b["kind"], (0.45, 0.31))
        if abs(d[0]) >= abs(d[1]):
            return c + [math.copysign(half[0], d[0]), 0]
        return c + [0, math.copysign(half[1], d[1])]

    def draw(self, path_no_ext, width=13.0):  # noqa: C901
        """Diagram with numbered callouts on the left, parameter table + symbol key on the right.
        Every value and source lives in the table, so nothing on the drawing can collide."""
        T.apply(); self._layout()
        # ---------------- rows for the parameter table
        rows = []        # (tag, title, [value lines], source)
        for k, b in self.bodies.items():
            if b["kind"] in ("ground", "point"):
                continue
            rows.append(("B", b["label"] or k, b["info"], ""))
        for l in self.links:
            if l["label"]:                                 # unlabeled links are plain wires (no row, no callout)
                rows.append(("E", l["label"], l["value"].split("\n") if l["value"] else [], l["source"]))
        for at, lab in self.probes:
            rows.append(("P", lab, [f"measured on {self.bodies[at]['label'] or at}"], "failure check"))
        LH = 0.19                                         # inches per table line
        def rl(r):
            return 1.25 + len(r[2]) + (r[3].count("\n") + 1 if r[3] else 0) + 0.75
        ncol = 3 if len(rows) > 4 else max(1, len(rows))
        total = sum(rl(r) for r in rows); cols = [[]]; acc = 0.0
        for r_, tg in zip(rows, [None] * len(rows)):
            if acc > total / ncol * 0.98 and len(cols) < ncol:
                cols.append([]); acc = 0.0
            cols[-1].append(r_); acc += rl(r_)
        col_lines = max(sum(rl(r) for r in c) for c in cols)
        xs_ = [b["pos"][0] for b in self.bodies.values()]; ys_ = [b["pos"][1] for b in self.bodies.values()]
        low = min(self.bodies.values(), key=lambda b: b["pos"][1])
        y_lo_pad = 0.45 if low["kind"] == "ground" else 1.55
        others = [b["pos"][1] for b in self.bodies.values() if b["kind"] != "ground"]
        y_lo = min(min(ys_) - y_lo_pad, min(others) - 1.55)
        lane_up0 = max([abs(l["lane"]) for l in self.links] + [0.0])
        xr = (max(xs_) + 1.4) - (min(xs_) - 1.2)
        yr = (max(ys_) + 1.1 + lane_up0) - (min(y_lo, -lane_up0 - 0.9) if lane_up0 else y_lo)
        head_in, key_in = 1.05, 0.85
        diag_in = min(max(0.94 * width * yr / xr, 1.2), 3.6)
        table_in = 0.35 + LH * col_lines
        foot_in = 0.2 + 0.17 * len(self.notes)
        height = head_in + diag_in + key_in + table_in + foot_in
        fig = plt.figure(figsize=(width, height))
        top = T.header(fig, self.title, self.subtitle, tag="SIMBLOCKS // PHYSICAL SCHEMATIC")
        y_diag = top - 0.1 / height - diag_in / height
        ax = fig.add_axes([0.03, y_diag, 0.94, diag_in / height]); ax.set_aspect("equal"); ax.axis("off")
        y_key = y_diag - key_in / height
        kax = fig.add_axes([0.03, y_key, 0.60, key_in / height * 0.9]); kax.axis("off")
        y_tab = y_key - table_in / height
        taxes = [fig.add_axes([0.03 + k * 0.94 / ncol, y_tab, 0.94 / ncol - 0.015, table_in / height]) for k in range(ncol)]
        for t_ in taxes:
            t_.axis("off"); t_.set_xlim(0, 1); t_.set_ylim(0, 1)
        fig.add_artist(matplotlib.lines.Line2D([0.03, 0.97], [y_key + 0.02 / height] * 2, color=T.EDGE, lw=0.6,
                                               transform=fig.transFigure, gid="qa-ignore"))
        xs = [b["pos"][0] for b in self.bodies.values()]; ys = [b["pos"][1] for b in self.bodies.values()]
        num = {"B": 0, "E": 0, "P": 0}; tags = []
        for r in rows:
            num[r[0]] += 1; tags.append(f"{r[0]}{num[r[0]]}")
        btag = {}; ti = 0
        for k, b in self.bodies.items():
            if b["kind"] in ("ground", "point"):
                continue
            btag[k] = tags[ti]; ti += 1
        nlab = sum(1 for l in self.links if l["label"])
        etag_l = iter(tags[ti: ti + nlab]); ptag = tags[ti + nlab:]
        etag = [next(etag_l) if l["label"] else None for l in self.links]
        # ---------------- elements with callouts only
        for l, tg in zip(self.links, etag):
            pa, pb = self.bodies[l["a"]]["pos"], self.bodies[l["b"]]["pos"]
            p1 = self._anchor(l["a"], pb); p2 = self._anchor(l["b"], pa)
            v = p2 - p1; nrm = np.array([-v[1], v[0]]) / max(np.hypot(*v), 1e-9)
            off = nrm * l["lane"]
            if l["lane"]:
                G.wire(ax, p1, p1 + off); G.wire(ax, p2, p2 + off)
            q1, q2 = p1 + off, p2 + off
            kind = l["kind"]
            if kind == "rot_elastoplastic":
                G.wire(ax, q1, q2); G.rot_elastoplastic(ax, (q1 + q2) / 2)
            elif kind == "rot_spring":
                G.wire(ax, q1, q2); G.rot_spring(ax, (q1 + q2) / 2)
            else:
                G.TWO_TERMINAL[kind](ax, q1, q2)
            m = (q1 + q2) / 2
            horiz = abs(v[0]) >= abs(v[1])
            c = m + ([0, 0.62] if horiz else [0.55, 0])
            if tg:
                _callout(ax, c, tg, T.ACCENT)
        # ---------------- bodies
        for k, b in self.bodies.items():
            c = b["pos"]; kind = b["kind"]
            if kind == "mass":
                G.mass(ax, c, w=_box_w(b["label"]), label=b["label"])
            elif kind == "rotor":
                G.rotor(ax, c)
            elif kind == "wheel":
                G.wheel(ax, c)
            elif kind == "motor":
                G.motor(ax, c)
            elif kind == "ground":
                G.ground(ax, c)
            elif kind == "face":
                G.hinge(ax, c)
            elif kind == "point":
                ax.plot([c[0]], [c[1]], "o", ms=5, color=T.INK, mec=T.SURFACE, mew=1.2, zorder=9)
            if kind in ("rotor", "wheel", "motor", "face") and b["label"]:
                below = any(k in (l["a"], l["b"]) and self.bodies[l["b"] if l["a"] == k else l["a"]]["pos"][1] < c[1] - 0.5
                            for l in self.links)
                # label goes on the side with no links, so a link never runs through it
                ax.text(c[0], c[1] + (0.78 if below else -0.78), b["label"], ha="center", va="bottom" if below else "top",
                        fontsize=8, color=T.INK, weight="semibold")
            if k in btag:
                grounded = any(k in (l["a"], l["b"]) and "ground" in (self.bodies[l["a"]]["kind"], self.bodies[l["b"]]["kind"])
                               for l in self.links)
                if grounded:
                    hw = _box_w(b["label"]) / 2 if kind == "mass" else 0.45
                    off = [-hw - 0.35, -0.55]
                else:
                    off = [0, -1.25 if kind in ("rotor", "wheel", "motor", "face") else -0.72]
                _callout(ax, np.array(c) + off, btag[k], T.INK_2)
        for (at, lab), tg in zip(self.probes, ptag):
            c = np.array(self.bodies[at]["pos"]) + [0.62, 0.55]
            G.probe(ax, c); ax.text(c[0] + 0.16, c[1], tg, color=T.ACCENT, fontsize=7.5, family=T.MONO,
                                    va="center", weight="bold")
        # room for a box's half width plus an offset callout beside it on either end of the row
        half = max([_box_w(b["label"]) / 2 for b in self.bodies.values() if b["kind"] == "mass"] + [0.45])
        lane_up = max([abs(l["lane"]) for l in self.links] + [0.0])   # links routed on lanes above/below the row
        ax.set_xlim(min(xs) - max(1.2, half + 1.0), max(xs) + max(1.4, half + 0.6))
        ax.set_ylim(min(y_lo, -lane_up - 0.9) if lane_up else y_lo, max(ys) + 1.1 + lane_up)
        # ---------------- parameter table (flows across columns)
        tagmap = {id(r_): tg for r_, tg in zip(rows, tags)}
        lh = LH / table_in
        for ci, (tax, col) in enumerate(zip(taxes, cols)):
            y = 1 - 0.35 / table_in * 0.9
            if ci == 0:
                tax.text(0, 1, "PARAMETERS", fontsize=7.5, color=T.ACCENT, family=T.MONO, weight="bold", va="top")
            for r_ in col:
                kind, title, vals, src = r_; tg = tagmap[id(r_)]
                colr = T.ACCENT if kind in ("E", "P") else T.INK_2
                tax.text(0, y, tg, fontsize=7.5, color=colr, family=T.MONO, weight="bold", va="top")
                tax.text(0.12, y, title, fontsize=8, color=T.INK, weight="semibold", va="top")
                y -= lh * 1.25
                for vline in vals:
                    tax.text(0.12, y, vline, fontsize=7, color=T.INK_2, family=T.MONO, va="top"); y -= lh
                if src:
                    tax.text(0.12, y, "src: " + src, fontsize=6.5, color=T.INK_3, family=T.MONO, va="top", linespacing=1.3)
                    y -= lh * (src.count("\n") + 1)
                y -= lh * 0.75
        self._key(kax)
        for i_, nt in enumerate(self.notes):
            fig.text(0.03, y_tab - (0.05 + 0.17 * i_) / height, nt, fontsize=6.5, color=T.INK_3, family=T.MONO, va="top")
        return T.save(fig, path_no_ext)

    def _key(self, kax):
        kax.set_xlim(0, 20); kax.set_ylim(-1.2, 1.0); kax.set_aspect("equal")
        kax.text(0, 0.75, "KEY", fontsize=7, color=T.ACCENT, family=T.MONO, weight="bold")
        items = [("spring", "spring"), ("elastoplastic", "spring + slider"), ("contact", "contact"),
                 ("dashpot", "damper"), ("rigid", "rigid link")]
        x = 0.0
        for k, lab in items:
            G.TWO_TERMINAL[k](kax, (x, -0.1), (x + 1.3, -0.1))
            kax.text(x + 0.65, -0.75, lab, fontsize=6.8, color=T.INK_2, ha="center", va="top")
            x += 2.6
        G.rot_elastoplastic(kax, (x + 0.4, -0.1), r=0.26); kax.text(x + 0.4, -0.75, "plastic hinge", fontsize=6.8, color=T.INK_2, ha="center", va="top")
        x += 2.4
        G.probe(kax, (x + 0.2, -0.1)); kax.text(x + 0.2, -0.75, "probe", fontsize=6.8, color=T.INK_2, ha="center", va="top")
        x += 1.8
        _callout(kax, (x + 0.2, -0.1), "E1", T.ACCENT); kax.text(x + 0.2, -0.75, "see table", fontsize=6.8, color=T.INK_2, ha="center", va="top")


def _box_w(label):
    """Mass box width that always contains its label (Inter 7.5 pt ~ 0.135 grid units per char)."""
    return max(0.9, 0.135 * len(label or "") + 0.45)


def _callout(ax, c, text, color):
    """Numbered callout chip."""
    from matplotlib.patches import FancyBboxPatch
    c = np.asarray(c, float)
    w = 0.17 * len(text) + 0.18; h = 0.34
    ax.add_patch(FancyBboxPatch((c[0] - w / 2, c[1] - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=T.SURFACE, ec=color, lw=1.0, zorder=10))
    ax.text(c[0], c[1], text, ha="center", va="center", fontsize=7, color=color, family=T.MONO, weight="bold", zorder=11)


class SignalFlow:
    """Per-step computation graph. stage(key, title, items=[(block, note)]) then edge(a, b, label)."""

    def __init__(self, title, subtitle=""):
        self.title, self.subtitle = title, subtitle
        self.stages = {}; self.edges = []; self.loops = []

    def stage(self, key, title, items=(), role="compute"):
        self.stages[key] = dict(title=title, items=list(items), role=role); return self

    def edge(self, a, b, label=""):
        self.edges.append((a, b, label)); return self

    def feedback(self, a, b, label=""):
        self.loops.append((a, b, label)); return self

    def draw(self, path_no_ext, size=(15.0, 4.7), tag="SIMBLOCKS // SIGNAL FLOW (ONE TIME STEP)"):
        """tag: header label; pass e.g. 'SIMBLOCKS // SEQUENCE' to draw a sequence of events instead of a step."""
        T.apply()
        fig = plt.figure(figsize=size)
        top = T.header(fig, self.title, self.subtitle, tag=tag)
        ax = fig.add_axes([0.02, 0.06, 0.96, top - 0.08]); ax.axis("off")
        n = len(self.stages); W, gap = 2.8, 1.15
        roles = {"state": T.ACCENT, "geometry": T.SERIES[0], "constitutive": T.SERIES[1], "force": T.SERIES[4],
                 "integrate": T.SERIES[2], "output": T.SERIES[3], "compute": T.EDGE}
        pos = {}
        maxitems = max(len(s["items"]) for s in self.stages.values()) if self.stages else 1
        H = 0.85 + 0.46 * maxitems
        for i, (k, s) in enumerate(self.stages.items()):
            x = i * (W + gap); pos[k] = x
            col = roles.get(s["role"], T.EDGE)
            ax.add_patch(FancyBboxPatch((x, -H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=T.PANEL, ec=T.EDGE, lw=1.0, zorder=2))
            ax.add_patch(FancyBboxPatch((x, H / 2 - 0.5), W, 0.5, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=T.PANEL_2, ec=T.EDGE, lw=1.0, zorder=3))
            ax.add_patch(plt.Rectangle((x, H / 2 - 0.5), 0.06, 0.5, color=col, zorder=4, lw=0))
            ax.text(x + 0.16, H / 2 - 0.25, s["title"].upper(), va="center", fontsize=7.4, color=T.INK,
                    family=T.MONO, weight="bold", zorder=5)
            for j, it in enumerate(s["items"]):
                name, note = (it if isinstance(it, tuple) else (it, ""))
                y = H / 2 - 0.80 - 0.46 * j
                ax.text(x + 0.16, y, name, fontsize=6.8, color=T.INK_2, family=T.MONO, va="center", zorder=5)
                if note:
                    ax.text(x + 0.16, y - 0.21, note, fontsize=6.3, color=T.INK_3, va="center", zorder=5)
            # ports
            for px, pa in ((x, 180), (x + W, 0)):
                ax.add_patch(Polygon([[px, 0.07], [px + (0.09 if pa == 0 else -0.0), 0], [px, -0.07]],
                                     closed=True, fc=T.INK_3, lw=0, zorder=5))
        for a, b, lab in self.edges:
            x0, x1 = pos[a] + W, pos[b]
            ax.add_patch(FancyArrowPatch((x0 + 0.1, 0), (x1 - 0.02, 0), arrowstyle="-|>", mutation_scale=9,
                                         color=T.INK_2, lw=1.1, zorder=6))
            if lab:
                ax.text((x0 + x1) / 2 + 0.05, 0.13, lab, ha="center", va="bottom", fontsize=6.4, color=T.INK_3,
                        family=T.MONO, rotation=0, zorder=6)
        for a, b, lab in self.loops:
            x0, x1 = pos[a] + W / 2, pos[b] + W / 2; yb = -H / 2 - 0.55
            pts = np.array([[x0, -H / 2], [x0, yb], [x1, yb], [x1, -H / 2]])
            ax.plot(pts[:3, 0], pts[:3, 1], color=T.ACCENT, lw=1.1, zorder=6)
            ax.add_patch(FancyArrowPatch(tuple(pts[2]), (x1, -H / 2 - 0.02), arrowstyle="-|>", mutation_scale=9,
                                         color=T.ACCENT, lw=1.1, zorder=6))
            ax.text((x0 + x1) / 2, yb - 0.12, lab, ha="center", va="top", fontsize=6.5, color=T.ACCENT, family=T.MONO)
        ax.set_xlim(-0.3, n * (W + gap)); ax.set_ylim(-H / 2 - 1.2, H / 2 + 0.3); ax.set_aspect("equal")
        return T.save(fig, path_no_ext)


@block("diagnostics", "diagram", "any_sim", "glyph_sheet", kind="builder", source="simblocks glyph library",
       tags=("diagram", "legend"))
def glyph_sheet(path_no_ext):
    """Render every glyph on one sheet (reference card for reading diagrams)."""
    T.apply()
    fig = plt.figure(figsize=(11, 4.6))
    top = T.header(fig, "Glyph library", "Standard lumped-parameter symbols used in every schematic",
                   tag="SIMBLOCKS // REFERENCE")
    ax = fig.add_axes([0.02, 0.04, 0.96, top - 0.06]); ax.set_aspect("equal"); ax.axis("off")
    items = [("spring", lambda: G.spring(ax, (0, 0), (1.6, 0))), ("dashpot", lambda: G.dashpot(ax, (2.4, 0), (4.0, 0))),
             ("plastic slider", lambda: G.slider(ax, (4.8, 0), (6.4, 0))),
             ("spring + slider", lambda: G.elastoplastic(ax, (7.2, 0), (9.2, 0))),
             ("contact gap", lambda: G.contact_gap(ax, (10.0, 0), (11.6, 0))),
             ("rigid link", lambda: G.rigid_link(ax, (12.4, 0), (14.0, 0)))]
    for lab, f in items:
        f()
    for x, lab in zip((0.8, 3.2, 5.6, 8.2, 10.8, 13.2), [i[0] for i in items]):
        ax.text(x, -0.55, lab, ha="center", fontsize=7.5, color=T.INK_2)
    row = [("mass", lambda c: G.mass(ax, c, label="m")), ("rotor", lambda c: G.rotor(ax, c)),
           ("motor", lambda c: G.motor(ax, c)), ("wheel", lambda c: G.wheel(ax, c)),
           ("hinge", lambda c: G.hinge(ax, c, r=0.12)), ("rot. spring", lambda c: G.rot_spring(ax, c)),
           ("plastic hinge", lambda c: G.rot_elastoplastic(ax, c)), ("ground", lambda c: G.ground(ax, c)),
           ("probe", lambda c: G.probe(ax, c))]
    for i, (lab, f) in enumerate(row):
        c = (0.8 + i * 1.65, -2.0)
        f(c); ax.text(c[0], -2.85, lab, ha="center", fontsize=7.5, color=T.INK_2)
    ax.set_xlim(-0.5, 14.6); ax.set_ylim(-3.2, 0.8)
    return T.save(fig, path_no_ext)

"""Glyph library: standard lumped-parameter symbols drawn as vector art in one consistent style.

Every two-terminal glyph has the signature  glyph(ax, p1, p2, color=..., **kw)  and draws between points
p1 and p2 (any orientation). One-point glyphs take a center. Sizes are in data units of an equal-aspect
axes where 1.0 = one grid cell. Standard rheology symbols are used so the diagram reads like a textbook:
spring = zigzag, dashpot = piston, plastic slider = St-Venant friction block, contact = gap with
facing plates, hinge = open pin, rotational spring = spiral.
"""
import math
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Arc, FancyArrowPatch, Wedge

from . import theme as T

LW = 1.6


def _frame(p1, p2):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    L = np.hypot(*d)
    u = d / L
    n = np.array([-u[1], u[0]])
    return p1, p2, u, n, L


def _line(ax, pts, color, lw=LW, z=3, **kw):
    pts = np.asarray(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z, **kw)


def wire(ax, p1, p2, color=T.INK_2, lw=1.2):
    _line(ax, [p1, p2], color, lw)


def spring(ax, p1, p2, color=T.SERIES[0], coils=5, width=0.18, lead=0.22):
    """Linear spring (zigzag)."""
    a, b, u, n, L = _frame(p1, p2)
    s0, s1 = a + u * L * lead, b - u * L * lead
    pts = [a, s0]
    m = coils * 2
    for i in range(1, m):
        t = i / m
        pts.append(s0 + (s1 - s0) * t + n * width * (1 if i % 2 else -1))
    pts += [s1, b]
    _line(ax, pts, color)


def dashpot(ax, p1, p2, color=T.SERIES[0], width=0.16):
    """Viscous damper (piston in cylinder)."""
    a, b, u, n, L = _frame(p1, p2)
    c0, c1 = a + u * L * 0.35, a + u * L * 0.68
    _line(ax, [a, c0 + u * 0.06], color)
    _line(ax, [c0 + n * width, c1 + n * width], color); _line(ax, [c0 - n * width, c1 - n * width], color)
    _line(ax, [c0 + n * width, c0 - n * width], color)
    pk = a + u * L * 0.55
    _line(ax, [pk + n * width * 0.75, pk - n * width * 0.75], color, lw=2.4)
    _line(ax, [pk, b], color)


def slider(ax, p1, p2, color=T.SERIES[1], width=0.16):
    """St-Venant plastic slider (friction block on a rail): yield element."""
    a, b, u, n, L = _frame(p1, p2)
    m = a + u * L * 0.5
    rail0, rail1 = m - u * L * 0.18 - n * width * 1.15, m + u * L * 0.18 - n * width * 1.15
    _line(ax, [rail0, rail1], color, lw=1.2)
    for k in range(5):
        q = rail0 + (rail1 - rail0) * (k / 4)
        _line(ax, [q, q - n * 0.07 - u * 0.06], color, lw=0.8)
    blk = [m - u * 0.11 - n * width, m + u * 0.11 - n * width, m + u * 0.11 + n * width * 0.5, m - u * 0.11 + n * width * 0.5]
    ax.add_patch(Polygon(blk, closed=True, fc=T.PANEL, ec=color, lw=LW, zorder=4))
    _line(ax, [a, m - u * 0.11], color); _line(ax, [m + u * 0.11, b], color)


def elastoplastic(ax, p1, p2, color_k=T.SERIES[0], color_p=T.SERIES[1]):
    """Spring in series with a plastic slider (Prandtl element): the standard symbol for a yielding spring."""
    a, b, u, n, L = _frame(p1, p2)
    m = a + u * L * 0.55
    spring(ax, a, m, color_k, coils=4, lead=0.15)
    slider(ax, m, b, color_p)


def contact_gap(ax, p1, p2, color=T.SERIES[4], gap=0.14, plate=0.22):
    """Unilateral contact: two facing plates with a gap (compression only)."""
    a, b, u, n, L = _frame(p1, p2)
    m = a + u * L * 0.5
    q0, q1 = m - u * gap / 2, m + u * gap / 2
    _line(ax, [a, q0], color); _line(ax, [q1, b], color)
    _line(ax, [q0 + n * plate, q0 - n * plate], color, lw=2.4)
    _line(ax, [q1 + n * plate, q1 - n * plate], color, lw=2.4)
    ax.annotate("", xy=q0 + n * plate * 1.4, xytext=q1 + n * plate * 1.4,
                arrowprops=dict(arrowstyle="<->", color=T.INK_3, lw=0.7, shrinkA=0, shrinkB=0))


def rigid_link(ax, p1, p2, color=T.INK_2):
    _line(ax, [p1, p2], color, lw=2.6)


def ground(ax, c, width=0.9, color=T.INK_3, angle=0.0):
    """Fixed support hatch centred at c (angle rotates the hatch; 0 = floor below)."""
    c = np.asarray(c, float); r = math.radians(angle)
    u = np.array([math.cos(r), math.sin(r)]); n = np.array([-u[1], u[0]])
    a, b = c - u * width / 2, c + u * width / 2
    _line(ax, [a, b], color, lw=1.6)
    for k in range(9):
        q = a + (b - a) * (k / 8)
        _line(ax, [q, q - n * 0.13 - u * 0.10], color, lw=0.8)


def hinge(ax, c, color=T.INK, r=0.08):
    """Revolute pin."""
    ax.add_patch(Circle(c, r, fc=T.PANEL, ec=color, lw=LW, zorder=6))


def rot_spring(ax, c, color=T.SERIES[0], r=0.26, turns=1.6):
    """Rotational (torsion) spring: spiral about c."""
    th = np.linspace(0, 2 * math.pi * turns, 200)
    rr = r * (0.25 + 0.75 * th / th[-1])
    c = np.asarray(c, float)
    _line(ax, np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)], color, lw=1.3)


def rot_elastoplastic(ax, c, color_k=T.SERIES[0], color_p=T.SERIES[1], r=0.3):
    """Rotational spring with a plastic hinge marker (filled half-disk = plastic hinge convention)."""
    rot_spring(ax, c, color_k, r=r)
    ax.add_patch(Wedge(c, r * 0.32, 90, 270, fc=color_p, ec=color_p, lw=0, zorder=7))
    ax.add_patch(Circle(c, r * 0.32, fc="none", ec=color_p, lw=1.2, zorder=7))


def mass(ax, c, w=0.9, h=0.62, color=T.INK, label=None):
    """Lumped mass / rigid body box."""
    c = np.asarray(c, float)
    ax.add_patch(FancyBboxPatch(c - [w / 2, h / 2], w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=T.PANEL_2, ec=color, lw=LW, zorder=4))
    if label:
        ax.text(*c, label, ha="center", va="center", color=T.INK, fontsize=7.5, family=T.SANS, weight="semibold", zorder=8)


def rotor(ax, c, R=0.42, color=T.SERIES[0], label=None, teeth=2):
    """Rotor / flywheel: disk with teeth and a spin arrow."""
    c = np.asarray(c, float)
    ax.add_patch(Circle(c, R, fc=T.PANEL_2, ec=color, lw=LW, zorder=4))
    for k in range(teeth):
        a = 2 * math.pi * k / teeth + 0.3
        tip = c + (R + 0.12) * np.array([math.cos(a), math.sin(a)])
        b1 = c + R * np.array([math.cos(a - 0.25), math.sin(a - 0.25)])
        b2 = c + R * np.array([math.cos(a + 0.12), math.sin(a + 0.12)])
        ax.add_patch(Polygon([b1, tip, b2], closed=True, fc=color, ec=color, lw=0.8, zorder=5))
    ax.add_patch(Arc(c, 2 * R * 0.62, 2 * R * 0.62, theta1=20, theta2=290, color=T.INK_2, lw=1.0, zorder=6))
    a = math.radians(290)
    ax.annotate("", xy=c + R * 0.31 * np.array([math.cos(a + 0.25), math.sin(a + 0.25)]),
                xytext=c + R * 0.31 * np.array([math.cos(a), math.sin(a)]),
                arrowprops=dict(arrowstyle="-|>", color=T.INK_2, lw=1.0, mutation_scale=7), zorder=6)
    hinge(ax, c, T.INK, 0.05)
    if label:
        ax.text(c[0], c[1] - R - 0.22, label, ha="center", va="top", color=T.INK, fontsize=7.5, weight="semibold")


def motor(ax, c, R=0.3, color=T.SERIES[3], label="M"):
    c = np.asarray(c, float)
    ax.add_patch(Circle(c, R, fc=T.PANEL_2, ec=color, lw=LW, zorder=4))
    ax.text(*c, label, ha="center", va="center", color=color, fontsize=10, family=T.MONO, weight="bold", zorder=5)


def wheel(ax, c, R=0.34, color=T.SERIES[2]):
    c = np.asarray(c, float)
    ax.add_patch(Circle(c, R, fc=T.PANEL_2, ec=color, lw=LW * 1.3, zorder=4))
    ax.add_patch(Circle(c, R * 0.55, fc="none", ec=T.EDGE, lw=0.8, zorder=5))
    hinge(ax, c, T.INK, 0.05)


def face(ax, root, length, angle_deg, color=T.INK, thick=0.07):
    """Hinged plate face (ramp, plow, panel) from root at angle (deg from +x, CCW)."""
    r = np.asarray(root, float); a = math.radians(angle_deg)
    tip = r + length * np.array([math.cos(a), math.sin(a)])
    _line(ax, [r, tip], color, lw=4.0, z=4)
    return tip


def probe(ax, c, color=T.ACCENT, label=None):
    """Measured output / failure-check probe (diamond)."""
    c = np.asarray(c, float); s = 0.09
    ax.add_patch(Polygon([c + [0, s], c + [s, 0], c + [0, -s], c + [-s, 0]], closed=True, fc=color, ec=T.SURFACE,
                         lw=1.0, zorder=9))
    if label:
        ax.text(c[0] + 0.14, c[1], label, color=color, fontsize=6.8, family=T.MONO, va="center", zorder=9)


def force_arrow(ax, p, d, color=T.STATUS["WARN"], label=None):
    p = np.asarray(p, float); d = np.asarray(d, float)
    ax.add_patch(FancyArrowPatch(p - d, p, arrowstyle="-|>", mutation_scale=10, color=color, lw=1.6, zorder=8))
    if label:
        q = p - d * 1.08
        ax.text(*q, label, color=color, fontsize=7, family=T.MONO, ha="center", va="center", zorder=8)


TWO_TERMINAL = {"spring": spring, "dashpot": dashpot, "slider": slider, "elastoplastic": elastoplastic,
                "contact": contact_gap, "rigid": rigid_link, "wire": wire}
